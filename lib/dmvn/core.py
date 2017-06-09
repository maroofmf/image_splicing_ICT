#!/usr/bin/env python2
"""
This file defines all required DMVN modules and layers

Created on Tue May 23 23:59:57 2017

@author: yue_wu
"""

#################################################################################
# Dependency
#################################################################################
import os
import sys
file_path = os.path.realpath( __file__ )
lib_root  = os.path.dirname( file_path )
model_root = os.path.join( lib_root, os.path.pardir, os.path.pardir, 'model' )
keras_lib = os.path.join( lib_root, os.path.pardir, 'keras_1.2.0/' )
assert os.path.isdir( keras_lib )
sys.path.insert( 0, keras_lib )
import keras
assert keras.__version__ == '1.2.0', "ERROR: only keras v1.2.0 is tested, but current version =%s" %  keras.__version__
from keras.models import Model
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.layers import Convolution2D,  MaxPooling2D, Flatten, UpSampling2D, Input, merge, Lambda
from theano import tensor as tt
from theano import scan
from keras.engine.topology import Layer
from keras.layers.pooling import GlobalAveragePooling2D

DEFAULT_DMVN_WEIGHT = os.path.join( model_root, 'dmvn_end_to_end.h5' )
assert os.path.isfile( DEFAULT_DMVN_WEIGHT ), "ERROR: failed to load default DMVN weight file %s" % DEFAULT_DMVN_WEIGHT
#################################################################################
# CNN Feature Extractor
#################################################################################
def create_vgg16_dense_kpf( input_shape, model_name = 'vgg16_dense_kpf', is_simplified = True ) :
    '''Create a CNN feature extractor that is identical to the well known VGG16 architecture
    INPUT:
        input_shape = tuple of ( nb_chs, nb_rows, nb_cols ), input image shape
        model_name  = string, name of the output model
        is_simplified = bool, whether or not use all 5 CNN blocks of VGG16
    OUTPUT:
        vgg16_cnn_featex = keras.model, output model
    '''
    img_input = Input( shape = input_shape )
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name= model_name + '-block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name= model_name + '-block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name= model_name + '-block1_pool')(x)
    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name= model_name + '-block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name= model_name + '-block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name= model_name + '-block2_pool')(x)
    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name= model_name + '-block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name= model_name + '-block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name= model_name + '-block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name= model_name + '-block3_pool')(x)
    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name= model_name + '-block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name= model_name + '-block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name= model_name + '-block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name= model_name + '-block4_pool')(x)
    # Block 5
    if ( not is_simplified ) :
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name= model_name + '-block5_conv1')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name= model_name + '-block5_conv2')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name= model_name + '-block5_conv3')(x)
    # Model
    return Model( input = img_input, output = x, name = model_name )

#################################################################################
# Deep Feature Correlation
#################################################################################
class TensorwiseCorrelation( Layer ) :
    '''Custom Tensorwise Correlation Layer
    '''
    def __init__( self, n_rows, n_cols, **kwargs ) :
        self.n_rows = n_rows
        self.n_cols = n_cols
        super( TensorwiseCorrelation, self ).__init__( **kwargs )
    def _get_mapping_indices( self ) :
        n_rows, n_cols = self.n_rows, self.n_cols
        new_indices = []
        for r_x in range( n_rows ) :
            for c_x in range( n_cols ) :
                for r_b in range( n_rows ) :
                    r_a = ( r_b + r_x ) % n_rows
                    for c_b in range( n_cols ) :
                        c_a = ( c_b + c_x ) % n_cols
                        idx_a = r_a * n_cols + c_a
                        idx_b = r_b * n_cols + c_b
                        idx = idx_a * ( n_rows * n_cols ) + idx_b
                        new_indices.append( idx )
        return new_indices
    def build( self, input_shape ) :
        self.midx = self._get_mapping_indices()
        self.built = True
    def call( self, tx_and_ty, mask = None ) :
        tx, ty = tx_and_ty
        n_samples, n_chs, n_rows, n_cols = tx.shape
        n_pixs = n_rows * n_cols
        txf = tx.reshape( ( n_samples, n_chs, n_pixs ) ).dimshuffle( 0, 2, 1 ) # ( n_samples, n_pixs, n_chs )
        tyf = ty.reshape( ( n_samples, n_chs, n_pixs ) ) / n_chs # ( n_samples, n_chs, n_pixs )
        res = tt.batched_dot( txf, tyf ) # ( n_samples, n_pixs, n_pixs )
        resf = res.flatten( 2 ).dimshuffle( 1, 0 )
        resm = resf[ self.midx ].dimshuffle( 1, 0 ).reshape( [n_samples, n_pixs, n_rows, n_cols] )
        return tt.cast( resm, 'float32' )
    def get_output_shape_for( self, input_shapes ) :
        tx_shape, ty_shape = input_shapes
        n_samples, n_chs, n_rows, n_cols = tx_shape
        output_shape = ( n_samples, n_rows * n_cols, n_rows, n_cols )
        return tuple( output_shape )

def create_model_corr_feat( input_shape = (256, 16, 16), model_name = 'dense_corr' ) :
    '''Create feature correlation model
    '''
    feat_x = Input( shape = input_shape, name = 'feat_x' )
    feat_y = Input( shape = input_shape, name = 'feat_y' )
    match  = TensorwiseCorrelation( n_rows = input_shape[1], n_cols = input_shape[2], name = 'match_xy' )( [feat_x, feat_y] )
    return Model( input = [ feat_x, feat_y ], output = match, name = model_name )

#################################################################################
# Correlation Feature Pooling
#################################################################################
def maxpooling_along_chs_opt( x ) :
    '''Max Pooling along the feature dimension
    '''
    return x.max( axis = 1, keepdims = True )

def maxpooling_along_chs_shape( input_shape ) :
    n_samples, n_chs, n_rows, n_cols = input_shape
    output_shape = ( n_samples, 1, n_rows, n_cols )
    return tuple( output_shape )

def avgpooling_along_chs_opt( x ) :
    '''Average Pooling along the feature dimension
    '''
    return x.mean( axis = 1, keepdims = True )

def avgpooling_along_chs_shape( input_shape ) :
    n_samples, n_chs, n_rows, n_cols = input_shape
    output_shape = ( n_samples, 1, n_rows, n_cols )
    return tuple( output_shape )

def argmaxpool_along_chs_opt_3d( t, topK = 6 ) :
    '''Argmax Pooling along the feature dimensions
    t = tensor4, nb_samples, nb_chs, nb_rows, nb_cols
    '''
    vals  = t.sum( axis = (-1,-2) )
    #index = vals.argmax( axis = -1 )
    index = vals.argsort( axis = -1 )[::-1][:topK]
    return t[index]

def argmaxpool_along_chs_opt( t, topK = 6 ) :
    '''t = tensor4, nb_samples, nb_chs, nb_rows, nb_cols
    '''
    t4d, _ = scan( fn = argmaxpool_along_chs_opt_3d, sequences = t )
    return t4d

def argmaxpool_along_chs_shape( input_shape, topK = 6 ) :
    n_samples, n_chs, n_rows, n_cols = input_shape
    output_shape = ( n_samples, topK, n_rows, n_cols )
    return tuple( output_shape )

def create_google_inception( name, input_dim, nb_inceptors, nb_filters = [1,3,5] ) :
    '''Create Google inception module
    '''
    x = Input( shape = ( input_dim, None, None), name = '%s_input' % name )
    icm = MaxPooling2D( (nb_filters[1], nb_filters[1] ), (1,1), border_mode = 'same', name = name +'_pool_inceptor_max' )( x )
    icm1x1 = Convolution2D( nb_inceptors, 1, 1, activation = 'relu', border_mode = 'same', name = name +'_pool_inceptor_max@1x1' )( icm )
    res = [ icm1x1 ]
    for idx, nf in enumerate( nb_filters ) :
        f1 = Convolution2D( nb_inceptors, 1, 1, activation = 'relu', border_mode = 'same', name = name +'_pool_inceptor_c%d@1x1' % idx )( x )
        if ( nf != 1 ) :
            cout = Convolution2D( nb_inceptors, nf, nf, activation = 'relu', border_mode = 'same', name = name +'_pool_inceptor_c%d@%d%d1' % (idx, nf, nf) )( f1 )
        else :
            cout = f1
        res.append( cout )
    ii  = merge( res, mode = 'concat', concat_axis = 1, name = 'pool_inceptor_concat' )
    inception = Model( input = [ x ], output = ii, name = name )
    return inception

def create_model_inception_matching_from_corr_paper( input_shape, nb_inceptors = 4, model_name = 'match_inception_corr', nb_filters = [1,3,5] ) :
    '''Create deep matching inception model
    '''
    match = Input( shape = input_shape, name = 'corr(x,y)' )
    ip1 = Lambda( function = maxpooling_along_chs_opt, output_shape = maxpooling_along_chs_shape, name = 'match_pool_p1' )( match )
    ip2 = Lambda( function = avgpooling_along_chs_opt, output_shape = maxpooling_along_chs_shape, name = 'match_pool_p2' )( match )
    ip3 = Lambda( function = argmaxpool_along_chs_opt, output_shape = argmaxpool_along_chs_shape, name = 'match_pool_p3' )( match )
    ff  = merge( [ ip1, ip2, ip3 ], mode = 'concat', concat_axis = 1, name = 'match_pool_concat' )
    pool_inceptor = create_google_inception( nb_inceptors = nb_inceptors, name = 'pool_inceptor', input_dim = 8, nb_filters = nb_filters )
    pf = pool_inceptor( ff )
    out = Convolution2D( nb_inceptors, 3, 3, activation = 'relu', border_mode = 'same', name = model_name+ '_inceptor_conv' )( pf )
    inception = Model( input = [match], output = out, name = model_name )
    return inception

#################################################################################
# Mask DeConvolution
#################################################################################
def create_inception_module( input_dim, nb_inceptors = 4, filt_dims = [ 1, 3, 5 ], name = 'inception', activation = 'relu' ) :
    tensor_input = Input( shape = ( input_dim, None, None ) )
    res_list = []
    for filt_size in filt_dims :
        res_list.append( Convolution2D( nb_inceptors, filt_size, filt_size, activation=activation, border_mode='same', name= '%s-filt@%dx%d' % ( name, filt_size, filt_size ) )( tensor_input ) )
    res = merge( res_list, mode = 'concat', concat_axis = 1 )
    return Model( input = tensor_input, output = res, name = name )

def create_model_mask_decorder( input_shape = ( 16, 16, 16 ),  nb_inceptors = 16, model_name = 'mask_prediction', is_single_input = False ) :
    if ( is_single_input ) :
        response = Input( shape = input_shape, name = 'merged_response' )
        model_input = response
        input_dim = input_shape[0]
    else :
        x_res_x = Input( shape = input_shape, name = 'x_res_x' )
        y_res_x = Input( shape = input_shape, name = 'y_res_x' )
        response = merge( [ x_res_x, y_res_x ], mode = 'concat', concat_axis = 1, name = 'merged_response' )
        model_input = [ x_res_x, y_res_x ]
        input_dim = input_shape[0] * 2
    # deconvolution 16x16 -> 32x32
    map32  = UpSampling2D( size = ( 2, 2 ), name = 'mask_map@32'  )( response )
    inception32 = create_inception_module( input_dim, nb_inceptors = 4, filt_dims = [ 1, 3, 5 ], name = 'mask_inceptor@32')
    res32  = inception32( map32 )
    # deconvolution 32x32 -> 64x64
    map64  = UpSampling2D( size = ( 2, 2 ), name = 'mask_map@64'  )( res32 )
    inception64 = create_inception_module( 12, nb_inceptors = 3, filt_dims = [ 1, 3, 5 ], name = 'mask_inceptor@64')
    res64  = inception64( map64 )
    # deconvolution 64x64 -> 128x128
    map128 = UpSampling2D( size = ( 2, 2 ), name = 'mask_map@128' )( res64 )
    inception128 = create_inception_module( 9, nb_inceptors = 2, filt_dims = [ 1, 3, 5 ], name = 'mask_inceptor@128')
    res128 = inception128( map128 )
    # deconvolution 64x64 -> 256x256
    map256 = UpSampling2D( size = ( 2, 2 ), name = 'mask_map@256' )( res128 )
    inception256 = create_inception_module( 6, nb_inceptors = 1, filt_dims = [ 1, 3, 5 ], name = 'mask_inceptor@256')
    res256 = inception256( map256 )
    # binary activation
    inceptionMask = create_inception_module( 3, nb_inceptors = 1, filt_dims = [ 5, 7, 11 ], name = 'mask_inceptorMask')
    mask256 = inceptionMask( res256 )
    mask = Convolution2D( 1, 1, 1, activation = 'sigmoid', border_mode = 'same', name = 'mask' )( mask256 )
    mask_pred = Model( input = model_input, output = mask, name = model_name )
    return mask_pred

#################################################################################
# Visual Consistency Validator
#################################################################################
def pool_mask( input_shape, name = 'maxpool_mask' ) :
    x = Input( shape = input_shape, name = name + '_mask' )
    x1 = MaxPooling2D((2,2), name = name + 'max_2' )(x)
    x2 = MaxPooling2D((2,2), name = name + 'max_4' )(x1)
    x3 = MaxPooling2D((2,2), name = name + 'max_8' )(x2)
    x4 = MaxPooling2D((2,2), name = name + 'max_16' )(x3)
    model = Model( input = x, output = x4, name = name )
    return model

thresh = 0.5
def visual_attention_option( t_tuple ) :
    feat, fmask = t_tuple
    bmask = tt.addbroadcast(fmask,1) > thresh
    return feat * bmask

def visual_attention_shape( input_shapes ) :
    feat_shape, mask_shape = input_shapes
    return feat_shape

def visual_attention( input_shape, name = 'visual_attention' ) :
    feat_in = Input( shape = input_shape, name = name + '_feat' )
    mask_in = Input( shape = tuple( [ 1 ] + list(input_shape[1:] ) ), name = name + '_mask' )
    vfeat = merge( inputs = [ feat_in, mask_in ], mode = visual_attention_option, output_shape = visual_attention_shape, name = name + '_opt' )
    model = Model( input = [ feat_in, mask_in ], output = vfeat, name = name )
    return model

def create_proba_featex( input_shape, name = 'proba_featex', nb_inceptors= 16 ) :
    x = Input( shape = input_shape, name = name + "_in" )
    vis_incetpor = create_google_inception( input_dim = input_shape[0], nb_inceptors = nb_inceptors, name = name + '_vis_inceptor' )
    x0 = vis_incetpor( x )
    x1 = MaxPooling2D((2,2), name = name + '_max_pool' )(x0)
    x2 = Convolution2D( nb_inceptors*2, 3, 3, activation = 'relu', border_mode = 'same', name = name + '_inceptor_conv' )( x1 )
    x3 = Flatten(name = name + "_flat")(x2)
    x4 = Dense( 1024, activation = 'relu', name = name + '_f1' )( x3 )
    x5 = Dropout( 0.5 )(x4)
    x6 = Dense( 512, activation = 'hard_sigmoid', name = name + '_f2' )( x5 )
    model = Model( input = x, output = x6, name = name )
    return model

def create_detection_branch( mask_shape = (1,256,256), feat_shape = (512,16,16), name = 'detection' ) :
    world_mask = Input( shape = mask_shape, name = name + '_world_mask')
    probe_mask = Input( shape = mask_shape, name = name + '_probe_mask')
    world_kpf = Input( shape = feat_shape, name = name + '_world_vgg')
    probe_kpf = Input( shape = feat_shape, name = name + '_probe_vgg')
    ####################################################################################
    # 6. visual attention mask
    focus_mask = pool_mask( mask_shape, 'pool_max' )
    world_att = focus_mask( world_mask )
    probe_att = focus_mask( probe_mask )
    concat_att = merge( [ world_att, probe_att ], mode = 'concat', concat_axis = 1, name = 'attention_concat' )
    att_feat = GlobalAveragePooling2D(name='global_pool')( concat_att )
    att_feat2 = Dense(8, activation = 'tanh', name = 'avg_pool_dense')( att_feat )
    # 7. visual attention ConvFeature
    attention = visual_attention( feat_shape )
    world_att_feat = attention( [ world_kpf, world_att ] )
    probe_att_feat = attention( [ probe_kpf, probe_att ] )
    # 8. visual attention DenseFeature
    proba_featex = create_proba_featex( feat_shape )
    world_pr_feat = proba_featex( world_att_feat )
    probe_pr_feat = proba_featex( probe_att_feat )
    # 9. siamese network
    diff = merge( [ world_pr_feat, probe_pr_feat ], mode = lambda x : x[0] - x[1], output_shape = lambda x : x[0], name = 'siamese' )
    drop = Dropout(0.5, name = 'dropOut@0.5')
    # siamese
    diff_f = drop( diff )
    proba_f = merge( [ diff_f, att_feat2 ], mode = 'concat', concat_axis = 1, name = 'proba_feat_concat' )
    diff_f = Dense( 128, activation = 'relu', name = 'siamese_f1' )( proba_f )
    diff_f = drop( diff_f )
    diff_f = Dense( 32, activation = 'relu', name = 'siamese_f2' )( diff_f )
    #
    proba_d = drop( diff_f )
    # 10. detection model
    detection = Model( input = [ world_mask, probe_mask, world_kpf, probe_kpf ], output = proba_d, name = name )
    return detection

#################################################################################
# End-to-End Deep Matching Validation Network
#################################################################################
def create_DMVN_model( image_input_shape = ( 3, 256, 256 ), pretrained_weight_file = DEFAULT_DMVN_WEIGHT ) :
    def _print_section_title( message ) :
        print "/" * 100
        print message
        print "/" * 100
        return
    #--------------------------------------------------------------------------------
    # define individual modules
    #--------------------------------------------------------------------------------
    # 1. create CNN Feature Extractor
    _print_section_title( "INFO: create CNN Feature Extractor" )
    featex = create_vgg16_dense_kpf( input_shape = image_input_shape, model_name = 'vgg16_cnn_featex', is_simplified = True )
    print featex.summary()
    feat_shape = tuple( list( featex.output_shape )[1:] )
    # 2. create Deep Dense Matching
    # 2.1 create Deep Feature Correlation module
    _print_section_title( "INFO: create Deep Feature Correlation module" )
    feat_corr = create_model_corr_feat( input_shape = feat_shape, model_name = 'dense_feat_corr' )
    print feat_corr.summary()
    corr_shape = feat_corr.output_shape[1:]
    # 2.2 create Corr Matching Pooling module
    corr_inceptor = create_model_inception_matching_from_corr_paper( corr_shape, 32, nb_filters = [1,3,5] )
    print corr_inceptor.summary()
    incetpor_shape = corr_inceptor.output_shape[1:]
    # 3. create Mask DeConvolution module
    _print_section_title( "INFO: create Mask DeConvolution module" )
    mask_pred = create_model_mask_decorder( input_shape = incetpor_shape, model_name = 'mask_predictor', is_single_input = False )
    print mask_pred.summary()
    mask_shape = mask_pred.output_shape[1:]
    # 4. create Visual Consistency Validator module
    _print_section_title( "INFO: create Visual Consistency Validator" )
    validator = create_detection_branch( mask_shape = mask_shape, feat_shape = feat_shape, name = 'visual_validator' )
    print validator.summary()
    #--------------------------------------------------------------------------------
    # assemble modules to an end-to-end DMVN
    #--------------------------------------------------------------------------------
    _print_section_title( "INFO: create end-to-end DMVN model")
    # 1. DMVN dual inputs
    world_img = Input( shape = image_input_shape, name = 'world' )
    probe_img = Input( shape = image_input_shape, name = 'probe' )
    # 2. CNN Feature Extractor
    world_kpf = featex( world_img )
    probe_kpf = featex( probe_img )
    self_corr_inceptor = corr_inceptor
    # 3. Deep Feature Correlation
    world_corr_probe = feat_corr( [ world_kpf, probe_kpf ] )
    probe_corr_world = feat_corr( [ probe_kpf, world_kpf ] )
    world_corr_world = feat_corr( [ world_kpf, world_kpf ] )
    probe_corr_probe = feat_corr( [ probe_kpf, probe_kpf ] )
    # 4. Correlation Pooling
    world_x_probe = corr_inceptor( world_corr_probe )
    probe_x_world = corr_inceptor( probe_corr_world )
    world_x_world = self_corr_inceptor( world_corr_world )
    probe_x_probe = self_corr_inceptor( probe_corr_probe )
    # 5. Mask Deconvolution
    world_mask = mask_pred( [ probe_x_world, world_x_world ] )
    probe_mask = mask_pred( [ world_x_probe, probe_x_probe ] )
    # 6. Visual Consistency Validator
    proba_feat = validator( [ world_mask, probe_mask, world_kpf, probe_kpf ] )
    # 7. Dual Tasks
    # 7.1 Splicing Localization
    masks = merge( [ world_mask, probe_mask ], concat_axis = 1, mode = 'concat', name = 'pred_masks' )
    # 7.2 Splicing Detection
    proba = Dense( 2, activation = 'softmax', name = 'pred_probs' )( proba_feat )
    # 7.3 End-to-End
    end_to_end = Model( input = [ world_img, probe_img ], output = [ masks, proba ], name = 'end_to_end_DMVN' )
    print end_to_end.summary()
    # 8. load weight file if necesssary
    if ( os.path.isfile( pretrained_weight_file ) ) :
        try :
            end_to_end.load_weights( pretrained_weight_file )
            _print_section_title( "INFO: successfully load pretrained weights from %s" % pretrained_weight_file )
        except Exception, e :
            print "WARNING: something wrong in loading pretrained weights from", pretrained_weight_file
            print "WARNING:", e
    return end_to_end
