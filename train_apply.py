'''Module to start normal training, validation and testing of ConvNet and UNet, 
hyperparameter optimization with both models and data augmentation tests.'''

from torch.utils.data import DataLoader

import params
import augment
import dataset
import models
import train_test

class DataSplit():
    '''
    Class that defines the dataloaders for train, validation, test split with the dataset class.
    
    Args:
        city_list (list): cities to include in the dataset
        train_size (int): length of the training set
        val_size (int): size of the validation set
        test_size (int): size of the test set
        batch_size (int): batch size
        augmentations (callable): augmentation or list of augmentations to apply

    Attriutes:
        train_loader (dict): dictionary of DataLoaders for each city of the training set
        val_loader (dict): dictionary of Dataloaders for each city of the test set
        test_loader (DataLoader): DataLoader for the test image
    '''

    def __init__(self, city_list=params.CITIES, train_size = params.TRAIN_SIZE, val_size = params.VAL_SIZE, test_size=params.TEST_SIZE, batch_size=params.BATCH_SIZE, augmentations=None):

        # create Dataloaders in dictinary for given lists of cities
        self.train_loader = {city:DataLoader(dataset.SegmentationDataset(city, 'training', dataset_size=train_size, augmentations=augmentations), batch_size, shuffle=True, num_workers=0) for city in city_list} # dictionary of dataloaders
        self.val_loader= {city:DataLoader(dataset.SegmentationDataset(city, 'validation', dataset_size=val_size), batch_size, shuffle=True, num_workers=0) for city in city_list} # dictionary of dataloaders
        self.test_loader = DataLoader(dataset.SegmentationDataset(params.TEST_CITY, 'test', dataset_size=test_size), batch_size, num_workers=0) # only dataloader!


def train_apply(model_name = None):
    '''
    Function for simple training and validation without hyperparameter optimization.
    '''

    band = 'all' # use all bands
    data_split = DataSplit() # init dataloader for train, valdiation, test

    # define model and its parameters
    if model_name == 'ConvNet':
        model = models.ConvNet(band, params.CONVNET_DROPOUT)
        train_output = params.CONVNET_SIMPLE_TRAIN
        val_output = params.CONVNET_SIMPLE_VAL
        learning_rates = params.CONVNET_LEARNING_RATES[0]
        l2_norm = params.CONVNET_L2_NORM[0]
        lr_sched = True
        iou_w = params.CONVNET_IOU_WEIGHT
        bce_w = params.CONVNET_BCE_WEIGHT

    elif model_name == 'UNet':
        model = models.UNet(band,params.OUT_DIM, params.UNET_DROPOUT)
        train_output = params.UNET_SIMPLE_TRAIN
        val_output = params.UNET_SIMPLE_VAL
        learning_rates = params.UNET_LEARNING_RATES[0]
        l2_norm = params.UNET_L2_NORM[0]
        lr_sched = False
        iou_w = params.UNET_IOU_WEIGHT
        bce_w = params.UNET_BCE_WEIGHT

    # move to device
    model = model.to(params.DEVICE)

    # start training and testing
    trainer = train_test.Trainer(
        model, 
        train_loader=data_split.train_loader, 
        val_loader=data_split.val_loader, 
        test_loader=data_split.test_loader, 
        train_output=train_output, 
        val_output=val_output, band=band, 
        weight_decay=l2_norm, 
        lr=learning_rates, 
        model_name=model.name,
        lr_scheduler=lr_sched,
        iou_w=iou_w,
        bce_w=bce_w
        )
    
    # change description
    trainer.description = f'{model_name}, bands: {band}, learning rate: {learning_rates}, weight decay: {l2_norm}'
    _ = trainer.training()
    _ = trainer.validation()
    _ = trainer.test()

def train_apply_hyper(model_name = None):
    '''
    Applies the hyperparameter optimization.
    '''
    if model_name == 'ConvNet':
        learning_rates = params.CONVNET_LEARNING_RATES
        l2_norm = params.CONVNET_L2_NORM
    elif model_name == 'UNet':
        learning_rates = params.UNET_LEARNING_RATES
        l2_norm = params.UNET_L2_NORM

    data_split = DataSplit() # init dataloader for train, valdiation, test

    # saves the models with hperparameters and performance for optimization
    performance = []
    # hyperparameter selection: channels
    for band in params.BANDS:
        # hyperparameter selection: dropout
        for lr in learning_rates:
            # hyperparameter selection: weight decays
            for weight_decay in l2_norm:

                # define model and its parameters
                if model_name == 'ConvNet':
                    model = models.ConvNet(band, params.CONVNET_DROPOUT)
                    train_output = params.CONVNET_TRAIN
                    val_output = params.CONVNET_VAL
                    iou_w = params.CONVNET_IOU_WEIGHT
                    bce_w = params.CONVNET_BCE_WEIGHT
                    lr_sched = True
                elif model_name == 'UNet':
                    model = models.UNet(band,params.OUT_DIM, params.UNET_DROPOUT)
                    train_output = params.UNET_TRAIN
                    val_output = params.UNET_VAL
                    iou_w = params.UNET_IOU_WEIGHT
                    bce_w = params.UNET_BCE_WEIGHT
                    lr_sched = False

                # move to device
                model = model.to(params.DEVICE)

                # start training and validation
                trainer = train_test.Trainer(
                    model, 
                    train_loader=data_split.train_loader, 
                    val_loader=data_split.val_loader, 
                    test_loader=data_split.test_loader, 
                    train_output=train_output, 
                    val_output=val_output, 
                    band=band, 
                    weight_decay=weight_decay, 
                    lr=lr, 
                    model_name=model.name, 
                    lr_scheduler=lr_sched,
                    iou_w=iou_w,
                    bce_w=bce_w)
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
    train_test.write_results(message, best_trainer.val_output)

    # run test
    f1 = best_trainer.test()

def augment_apply(model_name = None):
    '''
    Function for experiments with data augmentation.
    '''

    # list of single augmentations and a combination of augmentationss
    augmentations_dict = {'additive Gaussian noise':augment.add_gaussian_noise, 
                          'salt and pepper noise':augment.salt_pepper_noise, 
                          'horizontal flip':augment.h_flip, 
                          'random zoom':augment.rnd_zoom, 
                          'various noises': [augment.add_gaussian_noise, augment.salt_pepper_noise],
                          'zoom and Gaussian':[augment.rnd_zoom, augment.add_gaussian_noise]}

    # for all single augmentations and a combination of these augmentations
    for descr, augmentations in augmentations_dict.items():
        data_split = DataSplit(augmentations=augmentations) # init dataloader for train, valdiation, test

        # define model and its parameters
        if model_name == 'ConvNet':
            model = models.ConvNet(params.BAND, params.CONVNET_DROPOUT)
            train_output = params.CONVNET_AUG_TRAIN
            val_output = params.CONVNET_AUG_VAL
            learning_rates = params.CONVNET_LEARNING_RATES[0]
            l2_norm = params.CONVNET_L2_NORM[1]
            iou_w = params.CONVNET_IOU_WEIGHT
            bce_w = params.CONVNET_BCE_WEIGHT
            lr_sched = True

        elif model_name == 'UNet':
            model = models.UNet(params.BAND, params.OUT_DIM, params.UNET_DROPOUT)
            train_output = params.UNET_AUG_TRAIN
            val_output = params.UNET_AUG_VAL
            learning_rates = params.UNET_LEARNING_RATES[1]
            l2_norm = params.UNET_L2_NORM[0]
            iou_w = params.UNET_IOU_WEIGHT
            bce_w = params.UNET_BCE_WEIGHT
            lr_sched = False

            # move to device
            model = model.to(params.DEVICE)

        # start training and testing
        trainer = train_test.Trainer(
            model, 
            train_loader=data_split.train_loader, 
            val_loader=data_split.val_loader, 
            test_loader=data_split.test_loader, 
            train_output=train_output, 
            val_output=val_output, 
            band=params.BAND, 
            weight_decay=l2_norm, 
            lr=learning_rates, 
            model_name=model.name,
            lr_scheduler=lr_sched,
            iou_w=iou_w,
            bce_w=bce_w
            )
        
        # change description
        trainer.description = f'{model_name}, agumentation: {descr}'
        _ = trainer.training()
        _ = trainer.validation()
        _ = trainer.test()