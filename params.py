'''Module to define global parameters.'''

# data paths 
OSM_PATH = r'../OSM_maps/'
IMAGE_DATA_PATH = r'../image_data/'
DATASET_TRAIN = r'../dataset/training/'
DATASET_VAL = r'../dataset/validation/'
DATASET_TEST = r'../dataset/test/'
OUT_PATH = r'../output/'

CITIES = ['Aachen']
#CITIES = ['Aachen', 'Aarhus', 'Bonn','Copenhagen','Helsinki','Lausanne','Leipzig','Lyon','Porto','Potsdam'] 
TEST_CITY = 'Berlin'
TEST_COORDS = [13.294333, 52.454927, 13.500205, 52.574409] # (longitude west, latitude south, longitude east, latitude north)
#CITIES = ['test_pbf']

# dataset parameters
TRAIN_SIZE = 1280
VAL_SIZE = 320
BATCH_SIZE = 32
TEST_SIZE = 1
PATCH_SIZE = 128
BUILDING_COVER = 0.3 # default parameter for desired coverage of data with buildings
EPOCHS = 5

# ConvNet parameters
CONVNET_TRAIN = 'ConvNet_train_metrics' # train metrics output file name
CONVNET_VAL = 'ConvNet_val_metrics' # validation metrics output file name
CONVNET_AUG_TRAIN = 'ConvNet_train_augment_metrics' # train metrics augmentation 
CONVNET_AUG_VAL = 'ConvNet_test_augment_metrics' # validation metrics augmentation 
PRED_THRESHOLD = 0.5 # threshold for predicting a pixel as 'building'

# UNet parameters
UNET_TRAIN = 'UNet_train_metrics' # train metrics output file name
UNET_VAL = 'UNet_val_metrics' # validation metrics output file name
UNET_AUG_TRAIN = 'UNet_train_augment_metrics' # train metrics augmentation 
UNET_AUG_VAL = 'UNet_test_augment_metrics' # validation metrics augmentation 
OUT_DIM=1

# hyperparameters
DROPOUT = [0.2, 0.4, 0.5] # dropout rates
LEARNING_RATES = [1e-03, 1e-04, 1e-05] # learning rates
L2_NORM = [1e-3, 1e-4] # L2 normalization (weight decay)
BANDS = ['all','NIRGB', 'NIR'] # selection of channels

# augmentation parameters
BAND = 'all' # used bands for augmentation
PROB = 1 # prob. of augmentation being applied per sample
GMEAN = 0 # Gaussian mean
STDDEV = 0.07 # Gaussian standard deviation 
