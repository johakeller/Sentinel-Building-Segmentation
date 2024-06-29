# global parameters
# data paths 
OSM_PATH = r'../OSM_maps/'
IMAGE_DATA_PATH = r'../image_data/'
DATASET_TRAIN = r'../dataset/training/'
DATASET_VAL = r'../dataset/validation/'
DATASET_TEST = r'../dataset/test/'
OUT_PATH = r'../output/'

CITIES = ['Aachen', 'Aarhus', 'Bonn','Copenhagen','Helsinki','Lausanne','Leipzig','Lyon','Porto','Potsdam'] 
TEST_CITY = 'Berlin'
TEST_COORDS = [13.294333, 52.454927, 13.500205, 52.574409] # (longitude west, latitude south, longitude east, latitude north)
#CITIES = ['test_pbf']

# dataset parameters
TRAIN_SIZE = 1280
VAL_SIZE = 320
BATCH_SIZE = 32
TEST_SIZE = 1
PATCH_SIZE = 128
BUILDING_COVER = 0.04 # default parameter for desired coverage of data with buildings
EPOCHS = 1
LEARNING_RATE = 1e-03

# ConvNet parameters
CONVNET_TRAIN = 'ConvNet_train_metrics' # train metrics output file name
CONVNET_VAL = 'ConvNet_val_metrics' # validation metrics output file name
BAND = 'all' # which bands to use
PRED_THRESHOLD = 0.5
