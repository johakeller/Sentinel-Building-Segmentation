'''Module to define global parameters.'''

import torch

# data paths 
OSM_PATH = r'../OSM_maps/'
IMAGE_DATA_PATH = r'../image_data/'
DATASET_TRAIN = r'../dataset/training/'
DATASET_VAL = r'../dataset/validation/'
DATASET_TEST = r'../dataset/test/'
OUT_PATH = r'../output/'

#DATA_CITIES = ['Lyon']
DATA_CITIES = ['Aachen', 'Aarhus', 'Bonn','Copenhagen','Helsinki','Lausanne','Leipzig','Lyon','Porto','Potsdam'] # cities to collect data from
CITIES = ['Aachen', 'Bonn','Copenhagen'] # cities to perform training on
TEST_CITY = 'Berlin'
TEST_COORDS = [13.294333, 52.454927, 13.500205, 52.574409] # (longitude west, latitude south, longitude east, latitude north)

# dataset parameters
TRAIN_SIZE = 640
VAL_SIZE = 160
BATCH_SIZE = 64
TEST_SIZE = 1
PATCH_SIZE = 128
BUILDING_COVER = 0.3 # default parameter for desired coverage of data with buildings
EPOCHS = 5

# ConvNet parameters
CONVNET_TRAIN = 'ConvNet_hyper_train_metrics' # train metrics output file name
CONVNET_VAL = 'ConvNet_hyper_test_metrics' # validation metrics output file name
CONVNET_SIMPLE_TRAIN = 'ConvNet_train_metrics' # train metrics output file name (no hyperparameter optimization)
CONVNET_SIMPLE_VAL = 'ConvNet_test_metrics' # validation metrics output file name (no hyperparameter optimization)
CONVNET_AUG_TRAIN = 'ConvNet_train_augment_metrics' # train metrics augmentation 
CONVNET_AUG_VAL = 'ConvNet_test_augment_metrics' # validation metrics augmentation 

# ConvNet hyperparameters
CONVNET_DROPOUT = 0.25 # ConvNet
CONVNET_LEARNING_RATES = [1e-4, 5e-5] # ConvNet
CONVNET_L2_NORM = [1e-2, 5e-3] # ConvNet
CONVNET_CLASS_WEIGHT =torch.FloatTensor([1.0])

# UNet parameters 
UNET_TRAIN = 'UNet_hyper_train_metrics' # train metrics output file name
UNET_VAL = 'UNet_hyper_test_metrics' # validation metrics output file name
UNET_SIMPLE_TRAIN = 'UNet_train_metrics' # train metrics output file name (no hyperparameter optimization)
UNET_SIMPLE_VAL = 'UNet_test_metrics' # validation metrics output file name (no hyperparameter optimization)
UNET_AUG_TRAIN = 'UNet_train_augment_metrics' # train metrics augmentation 
UNET_AUG_VAL = 'UNet_test_augment_metrics' # validation metrics augmentation 
OUT_DIM = 1 # output

# UNet hyperparameters
UNET_DROPOUT = 0.1 # dropout rate
UNET_LEARNING_RATES = [5e-4, 1e-4, 5e-5] # learning rates
UNET_L2_NORM = [1e-3, 1e-4] # L2 normalization (weight decay)
UNET_CLASS_WEIGHT = torch.FloatTensor([2.33]) # weight for postive class

# global hyperparameters
BANDS = ['all','NIRGB', 'NIR'] # selection of channels
PRED_THRESHOLD = 0.5 # threshold for predicting a pixel as 'building'

# augmentation parameters
BAND = 'all' # used bands for augmentation
PROB = 0.5 # prob. of augmentation being applied per sample
GMEAN = 0 # Gaussian mean
STDDEV = 0.07 # Gaussian standard deviation 
