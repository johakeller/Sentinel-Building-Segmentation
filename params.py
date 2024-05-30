# global parameters
# data paths
OSM_PATH = r'/home/johakeller/Documents/Master Computer Science/Architecture of Machine Learning Systems/Exercise/OSM_maps/'
IMAGE_DATA_PATH = r'/home/johakeller/Documents/Master Computer Science/Architecture of Machine Learning Systems/Exercise/image_data/'
DATASET_PATH = r'/home/johakeller/Documents/Master Computer Science/Architecture of Machine Learning Systems/Exercise/dataset/'


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
BUILDING_COVER = 0.00 # default parameter for desired coverage of data with buildings
EPOCHS = 3
LEARNING_RATE = 1e-03

