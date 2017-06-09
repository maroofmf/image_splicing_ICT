# Deep Matching and Validation Network
#### An End-to-End Solution to Constrained Image Splicing Localization and Detection

The provided **python2** implementation of deep matching and validation network using the **Keras** deep neural network with the **Theano** backend. This repo is tested w.r.t. the following dependent libs. Libs of other versions are **not guaranteed** to work compatibly with the provided code. 

**Python**
- python 2.7.12 (better get Anaconda python https://www.continuum.io/downloads, and it will cover many of following dependencies)

**Matrix**
- numpy 1.12.1
- scipy 0.19.0

**Deep neural network**
- Keras  1.2  (already included https://github.com/fchollet/keras/tree/1.2.0)
  - yaml 3.12 (https://pypi.python.org/pypi/PyYAML)
  - h5py 2.6.0 (https://pypi.python.org/pypi/h5py/2.6.0)
- Theano 0.9 (Detailed installation instructions can be found at http://deeplearning.net/software/theano/install.html#install)
  Make sure you correctly set up your Keras config with the Theano backend!!! (see example in ./configs)

**Parallel processing**
- sklearn 0.18.1 (https://github.com/scikit-learn/scikit-learn/tree/0.18.X) 

**Image I/O dependency**
- OpenCV 2.4.9 (can be downloaded from https://github.com/opencv/opencv/tree/2.4.9)

**Plot dependency**
- matplotlib 1.5.1

## Directory Structure
- `config/` 
    - 'keras.json': a sample keras.json config with the Theano backend.
- `data/` 
    - `paired_CASIA_ids.csv`: defines the paired CASIA2 dataset
    - `README.md`: step-by-step instruction to prepare CASIA2 dataset
    - `small/`: contains 40 small RGB images from CASIA2
- `expt/` 
    - `test_on_paired_casia/`: contains DMVN prediction results on the paired CASIA2 dataset
- `lib/` 
    - `keras_1.2.0/`:  Keras lib
    - `dmvn/`: DMVN lib
- `model/` 
    - `dmvn_end_to_end.h5`: pretrained DMVN model
- `dmvn_example.ipynb`: ipython notebook of using DMVN to perform image splicing localization and detection using images from `data/small/`
- `dmvn_on_paired_casia.ipynb`: ipython notebook of testing DMVN performance on the paired CASIA2 dataset.
- `README.md`: the current file.

## Usage
Below is a simple code snippet of using the DMVN model to perform splicing localizaiton and detection on a pair of (probe, donor) images.

```python
# load DMVN model and image preprocess
from utils import preprocess_images
from core import create_DMVN_model

# create an end-to-end DMVN model
dmvn_end_to_end = create_DMVN_model()

# load two a DMVN sample of two images
Xp, Xd = preprocess_images( [ probe_file, donor_file ] )
X = { 'world' : Xd, 'probe' : Xp }

# splicing localization and detection via DMVN
pred_masks, pred_probs = dmvn_end_to_end.predict( X )
donor_mask, probe_mask = pred_masks[0]
splicing_prob = pred_probs.ravel()[1]
```

## Contact 
> Dr. Yue Wu

> Email: yue_wu@isi.edu

> Affiliation: USC Information Sciences Institute