import numpy as np
import torch

import params

#input is a torch tensor
# augmentations: change contrast randomly, gaussian noise, distort, flip vertical  -> with a certain percentage

def add_gaussian_noise(data_sample, probability=params.PROB, mean=params.GMEAN, sigma = params.STDDEV):

    # extract images from data sample
    bands = ['all', 'NIRGB', 'RGB', 'R', 'G', 'B', 'NIR']
    augmented_sample = {}

    # arbitrary number in [0,1]
    rndm = np.random.rand()

    # apply only to defined percentage of images
    if rndm <= probability:
        for band in bands:
            # extract each band from the sample
            image = data_sample[band].copy()

            image += np.random.normal(mean, sigma, image.shape)

            # clip to normal intensity range
            image = np.clip(image,0,255)

            # insert back into data_sample dictionary
            data_sample[band] = image

    return data_sample

def apply_augment(data_sample, augmentations):
    '''
    Create pipeline of augmentation functions and apply them to the input
    '''
    output = data_sample
    # check if a list of augmentations is passed
    if isinstance(augmentations, list):
        for augmentation in augmentations:
            output = augmentation(output)
        return output
    
    # only single augmentation is passed
    return augmentations(output)
