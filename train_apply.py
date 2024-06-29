from dataset import SegmentationDataset
from torch.utils.data import DataLoader

from models import *
from params import *
from data_acquisition import *
from train_test import *

# what dataset is loaded, how many bands, how to train etc.?
class DataSplit():
    def __init__(self, city_list=CITIES, train_size = TRAIN_SIZE, val_size = VAL_SIZE, test_size=TEST_SIZE, batch_size=BATCH_SIZE, patch_size = PATCH_SIZE, building_cover = BUILDING_COVER, augmentation=None):
        # create Dataloaders in dictinary for given lists of cities
        self.train_loader = {city:DataLoader(SegmentationDataset(city, 'training', dataset_size=train_size), batch_size, shuffle=True) for city in city_list} # dictionary of dataloaders
        self.val_loader= {city:DataLoader(SegmentationDataset(city, 'validation', dataset_size=val_size), batch_size, shuffle=True) for city in city_list} # dictionary of dataloaders
        self.test_loader = DataLoader(SegmentationDataset(TEST_CITY, 'test', dataset_size=test_size), batch_size) # only dataloader!


def train_apply(method=None):
    dataset = DataSplit() # init dataloader for train, valdiation, test
    models = [UNet(BAND,OUT_DIM), ConvNet(BAND)] # define models
    for model in models:
        trainer = Trainer(model, train_loader=dataset.train_loader, train_output=CONVNET_TRAIN, val_output=CONVNET_VAL)
        trainer.training()
        trainer.validation()

