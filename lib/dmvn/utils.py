#!/usr/bin/env python2
"""
This file defines all util functions for DMVN decoding

Created on Tue May 24 18:23:27 2017

@author: yue_wu
"""

import numpy as np
import cv2
import os
import sys
from copy import deepcopy
file_path = os.path.realpath( __file__ )
lib_root  = os.path.dirname( file_path )
keras_lib = os.path.join( lib_root, os.path.pardir, 'keras_1.2.0/' )
assert os.path.isdir( keras_lib )
sys.path.insert( 0, keras_lib )
import keras
assert keras.__version__ == '1.2.0', "ERROR: only keras v1.2.0 is tested, but current version =%s" %  keras.__version__
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as keras_image
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def convert_image_array_to_input_tensor( img ) :
    '''Convert an image uint8 array to float32 tensor4D
    INPUT:
        img = np.ndarray, dtype uint8, shape of ( height, width, 3 )
    OUTPUT:
        x   = np.ndarray, dtype float32, shape of ( 1, 3, height, width )
    NOTE:
        x is already normalized by subtracting image mean
    '''
    x = keras_image.img_to_array(img)
    x = np.expand_dims( x, axis = 0 )
    x = preprocess_input( x )
    return x

def load_one_image_robust( file_path, target_shape = (256, 256) ) :
    '''Load one image from file path
    INPUT:
        file_path = string, path to image file
        target_shape = tuple of two ints, ( nb_rows, nb_cols ) in output dims
    OUTPUT:
        tensor = np.ndarry, shape of ( 1, 3, nb_rows, nb_cols ) if succeed
                 None, otherwise
    '''
    if ( not os.path.isfile( file_path ) ) :
        return None
    try :
        img = cv2.imread( file_path, -1 )
        if ( img.shape[:2] != target_shape ) :
            img = cv2.resize( img, target_shape )
        if ( img.ndim == 2 ) :
            img = np.dstack( [ img for k in range(3) ] )
        if ( img.ndim > 3 ) :
            img = img[:,:,:3]
        tensor = convert_image_array_to_input_tensor( img )
        return tensor
    except Exception, e :
        print "ERROR: fail to load image", file_path, e
        return None

def load_splicing_input_csv( input_splicing_csv_file ) :
    '''Load input csv file for splicing detection and localization
    INPUT:
        input_splicing_csv_file = string, file path to input splicing csv file
    OUTPUT:
        splicing_pair_list = list, each elem is ( probe_image_file, donor_image_file )
        label_list = list, each elem is a bool indicating ground truth detection label
                     if no `label` column is found, then return None
    '''
    if ( not os.path.isfile( input_splicing_csv_file ) ) :
        raise IOError, "ERROR: cannot locate input splicing task csv file %s" % input_splicing_csv_file
    with open( input_splicing_csv_file, 'r' ) as IN :
        lines = [ line.strip() for line in IN.readlines() ]
    headers = [ h.lower() for h in lines.pop(0).split(',') ]
    assert ( 'probe' in headers ) and ( 'donor' in headers ), "ERROR: input splicing csv MUST have column of `probe` and `donor`"
    label_list = [] if ( 'label' in headers  ) else None
    splicing_pair_list = []
    for line in lines :
        fields = line.split(',')
        lut = dict( zip( headers, fields ) )
        splicing_pair_list.append( ( lut['probe'], lut['donor' ] ) )
        if ( label_list is not None ) :
            label_list.append( lut['label'] )
    return splicing_pair_list, label_list

def get_unique_images_and_idxLUT( splicing_image_pair_list ) :
    '''Get unique images and index look-up-table pairs from input splicing image pairs
    INPUT:
        splicing_image_pair_list = list, each elem is ( probe_image_file, donor_image_file )
    OUTPUT:
        unique_image_list = list, each elem is image_file
        splicing_idx_pair_list = list, each elem is ( probe_idx, donor_idx )
                                 where unique_image_list[ probe_idx ] = probe_image_file
                                 and unique_image_list[ donor_idx ] = donor_image_file
    NOTE:
        this function is aiming to suppress nonunique image files in the entire splicing task
    '''
    idx = 0
    unique_image_list, splicing_idx_pair_list = [], []
    seen = dict()
    for probe_file, donor_file in splicing_image_pair_list :
        if ( not seen.has_key( probe_file ) ) :
            seen[ probe_file ] = idx
            unique_image_list.append( probe_file )
            idx += 1
        if ( not seen.has_key( donor_file ) ) :
            seen[ donor_file ] = idx
            unique_image_list.append( donor_file )
            idx += 1
        # update idx_pair
        probe_idx, donor_idx = seen[ probe_file ], seen[ donor_file ]
        splicing_idx_pair_list.append( ( probe_idx, donor_idx ) )
    return splicing_idx_pair_list, unique_image_list

def preprocess_images( image_list, n_jobs = 1, verbose = 0 ) :
    '''Preprocess a list of image files to a list of DMVN input tensors
    INPUT:
        image_list = list or string,
                     if list, each elem is a string of file path to an image
                     if string, file path to image
        n_jobs = int, number of parallel jobs
        verbose = int, frequency level for reporting processing progress
    OUTPUT:
        tensor_list = list of tensors, each elem is a tensor4D or None
                      where None indicates failure
    '''
    if isinstance( image_list, str ) :
        image_list = [ image_list ]
    tensor_list = Parallel( n_jobs = 10, backend = 'threading', verbose = verbose )( delayed( load_one_image_robust )( fp ) for fp in image_list )
    return tensor_list

def filter_invalid_data( tensor_list, splicing_idx_pair_list, splicing_image_pair_list ) :
    '''Filter invalid image data
    INPUT:
        tensor_list = list of tensor, a valid elem should not be None
        splicing_idx_pair_list = list of ( probe_file_idx, donor_file_idx )
        splicing_image_pair_list = list of ( probe_file_path, donor_file_path )
    OUTPUT:
        valid_idx_pair_list = list of ( probe_file_idx, donor_file_idx )
        valid_image_pair_list = list of ( probe_file_path, donor_file_path )
    NOTE:
        only samples containing None tensor are filtered
    '''
    invalid_file_idx = [ k for k, t in enumerate( tensor_list ) if t is None ]
    nb_invalid = len( invalid_file_idx )
    print "INFO: meet #invalid_image_samples =", nb_invalid
    # terminate filter if all valid
    if ( nb_invalid == 0 ):
        return deepcopy( splicing_idx_pair_list ), deepcopy( splicing_image_pair_list )
    # filter invalid samples
    invalid_file_lut = dict( zip( invalid_file_idx, np.ones( nb_invalid ) ) )
    valid_idx_pair_list, valid_image_pair_list = [], []
    for sample_idx, probe_donor_idx_pair in enumerate( splicing_idx_pair_list ) :
        probe_idx, donor_idx = probe_donor_idx_pair
        if ( invalid_file_lut.has_key( probe_idx ) ) or ( invalid_file_lut.has_key( donor_idx ) ) :
            continue
        valid_idx_pair_list.append( probe_donor_idx_pair )
        valid_image_pair_list.append( splicing_image_pair_list[ sample_idx ] )
    print "INFO: filter out", len( splicing_idx_pair_list) - len( valid_idx_pair_list ), "invalid DMVN samples"
    return valid_idx_pair_list, valid_image_pair_list

def prepare_one_batch_input( tensor_list, batch_idx_pair_list ) :
    '''Prepare image tensors for DMVN input tuples
    INPUT:
        tensor_list = list of tensor, each elem is an image tensor
        batch_idx_pair_list = list of ( probe_file_idx, donor_file_idx )
    OUTPUT:
        X = dict, { 'probe'-> probe_tesnor, 'world'-> donor_tensor }
    '''
    X_probe, X_donor = [], []
    for probe_idx, donor_idx in batch_idx_pair_list :
        X_probe.append( tensor_list[ probe_idx ] )
        X_donor.append( tensor_list[ donor_idx ] )
    X_probe = np.concatenate( X_probe, axis = 0 )
    X_donor = np.concatenate( X_donor, axis = 0 )
    return { 'probe' : X_probe, 'world' : X_donor }

def restore_img_tensor( img_tensor ) :
    '''Restore an DMVN input float32 tensor to uint8 RGB image
    INPUT:
        img_tensor = np.ndarray, shape of ( 3, 256, 256 ), dtype = float32
    OUTPUT:
        img_array  = np.ndarray, shape of ( 256, 256, 3 ), dtype = uint8
    '''
    img_tensor[0] += 103.939
    img_tensor[1] += 116.779
    img_tensor[2] += 123.680
    return np.rollaxis( img_tensor[::-1], 0, 3 ).astype('uint8')

def visualize_debug_dualmask( world_tensor, probe_tensor, pred_mask, pred_pr, output_file = 'vis-decode.jpg', th = -1, visualization_level = -1 ) :
    '''Visualize one DMVN sample to output_file
    INPUT:
        world_tensor = tensor3D, shape of ( 3, 256, 256 ), dtype = float32
        probe_tensor = tensor3D, shape of ( 3, 256, 256 ), dtype = float32
        pred_mask = tensor3D, shape of ( 2, 256, 256 ), dtype = float32, output of DMVN's $pred_masks
        pred_pr = tensor2D, shape of ( 2, 2 ), dtype = float32, output of DMVN's $pred_probs
        output_file = string, file path to the output file
        th = float, threshold to binary predicted masks, if negative then use grayscale masks
        visualization_level = int, <= -1: suppress all; =0: only print out proba; =1: only save positive predictions; >=2 save everything
    OUTPUT:
        succeess = bool, whether or not normally exit this visualization process
    '''
    def restore_mask_tensor( mask_tensor ) :
        mask_tensor[ mask_tensor > 1 ] = 1
        mask_tensor[ mask_tensor < 0 ] = 0
        if ( th > 0 ) and ( th < 1 ) :
            mask_tensor[ mask_tensor >= th ] = 1
            mask_tensor[ mask_tensor < th ]  = 0
        mask_tensor = ( mask_tensor * 255 ).astype('uint8')
        mask_image = np.column_stack( [ mask_tensor[0], mask_tensor[1] ] )
        return np.dstack( [ mask_image, mask_image, mask_image ] )
    def add_proba_image( pred_pr ) :
        score = np.ones( ( 256, 256, 3 ) ).astype('uint8') * 127
        perc_w = ( ( pred_mask[0] > th ).sum() / 65536. )
        perc_p = ( ( pred_mask[1] > th ).sum() / 65536. )
        cv2.putText( score, 'Perc_w = %.2f' % perc_w,  (3,90), fontScale = 1, fontFace = 0, color = ( 255, 0, 0 ), thickness = 3 )
        cv2.putText( score, 'Perc_p = %.2f' % perc_p,  (3,150), fontScale = 1, fontFace = 0, color = ( 0, 255, 0 ), thickness = 3 )
        cv2.putText( score, 'Proba. = %.2f' % pred_pr[1], (3,210), fontScale = 1, fontFace = 0, color = ( 0, 0, 255 ), thickness = 3 )
        return score, [ pred_pr[1], pred_pr[-1], perc_w, perc_p ]
    if ( visualization_level < 0 ) :
        return True
    if ( visualization_level == 0 ) :
        print "pred_proba =", pred_pr[1]
        return True
    print output_file, "pred_proba =", pred_pr[1]
    world = restore_img_tensor( world_tensor.copy() )
    probe = restore_img_tensor( probe_tensor.copy() )
    col1 = np.column_stack( [ world, probe ] )
    col2 = restore_mask_tensor( pred_mask.copy() )
    col3 = ( col1.astype( np.float32 ) * .6 ).astype( 'uint8' ) + ( col1.astype( np.float32 ) * .4 * ( col2 > 127 ) ).astype( 'uint8' )
    simg, score_list = add_proba_image( pred_pr )
    debug = np.column_stack( [ col1, col2, col3, simg ] )
    save_debug = True if ( ( visualization_level >=2 ) or ( pred_pr[1] > .5 ) ) else False
    if ( save_debug ) :
        return cv2.imwrite( output_file, debug )
    else :
        return True

def evaluate_fscore( y_true, y_pred_bool, system_name = 'default' ) :
    '''Evaluate system performance using precision, recall, and fscore
    '''
    p, r, f, s = precision_recall_fscore_support( y_true, y_pred_bool, average = 'binary' )
    print "INFO:", system_name, "fscore =", f
    return f

def evaluate_auc( y_true, y_pred_float, system_name = 'default' ) :
    '''Evaluate system performance using area under curve
    '''
    auc = roc_auc_score( y_true, y_pred_float )
    print "INFO:", system_name, "auc =", auc
    return auc
    
def visualize_ipynb(  world_tensor, probe_tensor, pred_mask, pred_pr, th = -1 ) :
    '''Visualize one DMVN sample in ipynb
    INPUT:
        world_tensor = tensor3D, shape of ( 3, 256, 256 ), dtype = float32
        probe_tensor = tensor3D, shape of ( 3, 256, 256 ), dtype = float32
        pred_mask = tensor3D, shape of ( 2, 256, 256 ), dtype = float32, output of DMVN's $pred_masks
        pred_pr = tensor2D, shape of ( 2, 2 ), dtype = float32, output of DMVN's $pred_probs
        th = float, threshold to binary predicted masks, if negative then use grayscale masks
    OUTPUT:
        vis_input = np.ndarray, shape of ( 256, 512, 3 ), dtype = uint8, visualization of inputs
        vis_mask = np.ndarray, shape of ( 256, 512, 3 ), dtype = uint8, visualization of masks
        vis_overlay = np.ndarray, shape of ( 256, 512, 3 ), dtype = uint8, visualization of overlaied
    '''
    def restore_mask_tensor( mask_tensor ) :
        mask_tensor[ mask_tensor > 1 ] = 1
        mask_tensor[ mask_tensor < 0 ] = 0
        if ( th > 0 ) and ( th < 1 ) :
            mask_tensor[ mask_tensor >= th ] = 1
            mask_tensor[ mask_tensor < th ]  = 0
        mask_tensor = ( mask_tensor * 255 ).astype('uint8')
        mask_image = np.column_stack( [ mask_tensor[0], mask_tensor[1] ] )
        return np.dstack( [ mask_image, mask_image, mask_image ] )
    world = restore_img_tensor( world_tensor.copy() )
    probe = restore_img_tensor( probe_tensor.copy() )
    vis_input = np.column_stack( [ world, probe ] )
    vis_mask  = restore_mask_tensor( pred_mask.copy() )
    vis_overlay = ( vis_input.astype( np.float32 ) * .6 ).astype( 'uint8' ) + ( vis_input.astype( np.float32 ) * .4 * ( vis_mask > 127 ) ).astype( 'uint8' )
    return vis_input[:,:,::-1], vis_mask, vis_overlay[:,:,::-1]
