from dataset import SegmentationDataset
from torch.utils.data import DataLoader

import params
import augment
from models import *
from params import *
from data_acquisition import *
from train_test import *

class DataSplit():
    def __init__(self, city_list=CITIES, train_size = TRAIN_SIZE, val_size = VAL_SIZE, test_size=TEST_SIZE, batch_size=BATCH_SIZE, patch_size = PATCH_SIZE, building_cover = BUILDING_COVER, augmentations=None):
        # create Dataloaders in dictinary for given lists of cities
        self.train_loader = {city:DataLoader(SegmentationDataset(city, 'training', dataset_size=train_size, augmentations=augmentations), batch_size, shuffle=True, num_workers=0) for city in city_list} # dictionary of dataloaders
        self.val_loader= {city:DataLoader(SegmentationDataset(city, 'validation', dataset_size=val_size), batch_size, shuffle=True, num_workers=0) for city in city_list} # dictionary of dataloaders
        self.test_loader = DataLoader(SegmentationDataset(TEST_CITY, 'test', dataset_size=test_size), batch_size, num_workers=0) # only dataloader!


def train_apply_hyper(model_name = None):
    '''
    Applies the hyperparameter optimization
    '''
    if model_name == 'ConvNet':
        learning_rates = CONVNET_LEARNING_RATES
        l2_norm = CONVNET_L2_NORM
        dropout = CONVNET_DROPOUT
    elif model_name == 'UNet':
        learning_rates = UNET_LEARNING_RATES
        l2_norm = UNET_L2_NORM
        dropout = UNET_DROPOUT
    dataset = DataSplit() # init dataloader for train, valdiation, test
    # saves the models with hperparameters and performance for optimization
    performance = []
    # hyperparameter selection: channels
    for band in BANDS:
        # hyperparameter selection: dropout
        for lr in learning_rates:
            # hyperparameter selection: weight decays
            for weight_decay in l2_norm:

                # define model and its parameters
                if model_name == 'ConvNet':
                    model = ConvNet(band, dropout)
                    train_output = CONVNET_TRAIN
                    val_output = CONVNET_VAL
                    class_weight = CONVNET_CLASS_WEIGHT
                elif model_name == 'UNet':
                    model = UNet(band,OUT_DIM, dropout)
                    train_output = UNET_TRAIN
                    val_output = UNET_VAL
                    class_weight = UNET_CLASS_WEIGHT
                # start training and validation
                trainer = Trainer(
                    model, 
                    train_loader=dataset.train_loader, 
                    val_loader=dataset.val_loader, 
                    test_loader=dataset.test_loader, 
                    train_output=train_output, 
                    val_output=val_output, 
                    band=band, 
                    weight_decay=weight_decay, 
                    lr=lr, dropout=dropout, 
                    model_name=model.name, 
                    class_weight=class_weight, 
                    lr_scheduler=True)
                _ = trainer.training()
                f1 = trainer.validation()

                # insert performance into list
                performance.append([f1, trainer])
    
    # save best performance
    max_score = 0.0
    best_trainer = None

    # find best hyperparameters
    for value in performance:
        if value[0] > max_score:
            # update max_score
            max_score = value[0]
            # update trainer (contains model)
            best_trainer = value[1]
            
    # write and display hyperparameter info
    message = f'Hyperparameters selected with F1 score of {max_score:.2f}: ' + best_trainer.description +'\n'
    print(message)
    write_results(message, best_trainer.val_output)

    # run test
    f1 = best_trainer.test()

def augment_apply(model_name = None):
    '''
    Function for experiments with data augmentation.
    '''

    # list of single augmentations and a combination of augmentationss
    augmentations_dict = {#'additive Gaussian noise':augment.add_gaussian_noise, 
                          'salt and pepper noise':augment.salt_pepper_noise, 
                          'horizontal flip':augment.h_flip, 
                          'random zoom':augment.rnd_zoom, 
                          'various noise': [augment.add_gaussian_noise, augment.salt_pepper_noise],
                          'all augmentations':[augment.h_flip, augment.rnd_zoom, augment.salt_pepper_noise, augment.add_gaussian_noise]}

    # for all single augmentations and a combination of these augmentations
    for descr, augmentations in augmentations_dict.items():
        dataset = DataSplit(augmentations=augmentations) # init dataloader for train, valdiation, test

        # define model and its parameters
        if model_name == 'ConvNet':
            model = ConvNet(params.BAND, CONVNET_DROPOUT)
            train_output = CONVNET_AUG_TRAIN
            val_output = CONVNET_AUG_VAL
            learning_rates = CONVNET_LEARNING_RATES[1]
            l2_norm = CONVNET_L2_NORM[1]
            dropout = CONVNET_DROPOUT
            class_weight = CONVNET_CLASS_WEIGHT

        elif model_name == 'UNet':
            model = UNet(params.BAND,OUT_DIM, UNET_DROPOUT)
            train_output = UNET_AUG_TRAIN
            val_output = UNET_AUG_VAL
            learning_rates = UNET_LEARNING_RATES[1]
            l2_norm = UNET_L2_NORM[1]
            dropout = UNET_DROPOUT
            class_weight = UNET_CLASS_WEIGHT

        # start training and testing
        trainer = Trainer(
            model, 
            train_loader=dataset.train_loader, 
            val_loader=dataset.val_loader, 
            test_loader=dataset.test_loader, 
            train_output=train_output, 
            val_output=val_output, 
            band=params.BAND, 
            weight_decay=l2_norm, 
            lr=learning_rates, 
            dropout=dropout, 
            model_name=model.name,
            class_weight=class_weight,
            lr_scheduler=True
            )
        
        # change description
        trainer.description = f'{model_name}, agumentation: {descr}'
        _ = trainer.training()
        _ = trainer.validation()
        _ = trainer.test()

def train_apply(model_name = None):
    '''
    Function for simple training and validation without hyperparameter optimization.
    '''

    band = 'all' # use all bands
    dataset = DataSplit() # init dataloader for train, valdiation, test

    # define model and its parameters
    if model_name == 'ConvNet':
        model = ConvNet(band, CONVNET_DROPOUT, batch_norm=False)
        train_output = CONVNET_SIMPLE_TRAIN
        val_output = CONVNET_SIMPLE_VAL
        learning_rates = CONVNET_LEARNING_RATES[1]
        l2_norm = CONVNET_L2_NORM[1]
        dropout = 0.0 # no dropout
        class_weight = CONVNET_CLASS_WEIGHT

    elif model_name == 'UNet':
        model = UNet(band,OUT_DIM, UNET_DROPOUT)
        train_output = UNET_SIMPLE_TRAIN
        val_output = UNET_SIMPLE_VAL
        learning_rates = UNET_LEARNING_RATES[1]
        l2_norm = UNET_L2_NORM[1]
        dropout = 0.0 # no dropout
        class_weight = UNET_CLASS_WEIGHT

    # start training and testing
    trainer = Trainer(
        model, 
        train_loader=dataset.train_loader, 
        val_loader=dataset.val_loader, 
        test_loader=dataset.test_loader, 
        train_output=train_output, 
        val_output=val_output, band=band, 
        weight_decay=l2_norm, 
        lr=learning_rates, 
        dropout=dropout, 
        model_name=model.name,
        class_weight=class_weight
        )
    # change description
    trainer.description = f'{model_name}, bands: {band}, learning rate: {learning_rates}'
    _ = trainer.training()
    _ = trainer.validation()
    _ = trainer.test()








