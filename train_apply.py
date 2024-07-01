from dataset import SegmentationDataset
from torch.utils.data import DataLoader

import augment
from models import *
from params import *
from data_acquisition import *
from train_test import *

class DataSplit():
    def __init__(self, city_list=CITIES, train_size = TRAIN_SIZE, val_size = VAL_SIZE, test_size=TEST_SIZE, batch_size=BATCH_SIZE, patch_size = PATCH_SIZE, building_cover = BUILDING_COVER, augmentations=None):
        # create Dataloaders in dictinary for given lists of cities
        self.train_loader = {city:DataLoader(SegmentationDataset(city, 'training', dataset_size=train_size, augmentations=augmentations), batch_size, shuffle=True) for city in city_list} # dictionary of dataloaders
        self.val_loader= {city:DataLoader(SegmentationDataset(city, 'validation', dataset_size=val_size), batch_size, shuffle=True) for city in city_list} # dictionary of dataloaders
        self.test_loader = DataLoader(SegmentationDataset(TEST_CITY, 'test', dataset_size=test_size), batch_size) # only dataloader!


def train_apply(model_name = None):
    '''
    Applies the hyperparameter optimization
    '''
    dataset = DataSplit() # init dataloader for train, valdiation, test
    # saves the models with hperparameters and performance for optimization
    performance_dict = {}
    # hyperparameter selection: channels
    for band in BANDS:
        # hyperparameter selection: dropout
        for lr in LEARNING_RATES:
            # hyperparameter selection: weight decays
            for weight_decay in L2_NORM:

                # define model and its parameters
                if model_name == 'ConvNet':
                    model = ConvNet(band, DROPOUT[0])
                    train_output = CONVNET_TRAIN
                    val_output = CONVNET_VAL
                elif model_name == 'UNet':
                    model = UNet(band,OUT_DIM, DROPOUT[0])
                    train_output = UNET_TRAIN
                    val_output = UNET_VAL
                # start training and testing
                trainer = Trainer(model, train_loader=dataset.train_loader, val_loader=dataset.val_loader, test_loader=dataset.test_loader, train_output=train_output, val_output=val_output, band=band, weight_decay=weight_decay, lr=lr, dropout=DROPOUT[0], model_name=model.name)
                _ =trainer.training()
                f1 = trainer.validation()
                # insert performance into dictionary
                performance_dict[trainer.description]= [f1, trainer]
    
    # find best hyperparameters
    max_score = 0.0
    for value in performance_dict.values():
        if value[0] > max_score:
            # obtain trainer
            trainer = value[1]
            
            # write and display hyperparameter info
            message = f'Hyperparameters selected with F1 score of {value[0]:.2f}: ' + trainer.description
            print(message)
            write_results(message, trainer.val_output)

            # run test
            f1 = trainer.test()

def augment_apply(model_name = None):
    '''
    Function for experiments with data augmentation.
    '''

    # list of single augmentations and a combination of augmentationss
    augmentations_dict = {'additive Gaussian noise':augment.add_gaussian_noise, 
                          'random contrast':augment.rnd_contrast, 
                          'horizontal flip':augment.h_flip, 
                          'random zoom':augment.rnd_zoom, 
                          'all augmentations':[augment.h_flip, augment.rnd_zoom, augment.rnd_contrast, augment.add_gaussian_noise]}

    # for all single augmentations and a combination of these augmentations
    for descr, augmentations in augmentations_dict.items():
        band = 'all' # use all bands
        dataset = DataSplit(augmentations=augmentations) # init dataloader for train, valdiation, test

        # define model and its parameters
        if model_name == 'ConvNet':
            model = ConvNet(band, DROPOUT[0])
            train_output = CONVNET_AUG_TRAIN
            val_output = CONVNET_AUG_VAL
        elif model_name == 'UNet':
            model = UNet(band,OUT_DIM, DROPOUT[0])
            train_output = UNET_AUG_TRAIN
            val_output = UNET_AUG_VAL
        # start training and testing
        trainer = Trainer(model, train_loader=dataset.train_loader, val_loader=dataset.val_loader, test_loader=dataset.test_loader, train_output=train_output, val_output=val_output, band=band, weight_decay=L2_NORM[1], lr=LEARNING_RATES[1], dropout=DROPOUT[0], model_name=model.name)
        # change description
        trainer.description = f'{model_name}, agumentation: {descr}'
        _ =trainer.training()
        _ = trainer.validation()











