import numpy as np
import cv2

import params

def salt_pepper_noise(data_sample, probability=params.PROB, mean=params.GMEAN, sigma = params.STDDEV):

    # extract images from data sample
    bands = [params.BAND]

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

def add_gaussian_noise(data_sample, probability=params.PROB, mean=params.GMEAN, sigma = params.STDDEV):

    # extract images from data sample
    bands = [params.BAND]

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

def rnd_contrast(data_sample, probability=params.PROB):

    # extract images from data sample
    bands = [params.BAND]

    # arbitrary number in [0,1]
    rndm = np.random.rand()

    # apply only to defined percentage of images
    if rndm <= probability:
        for band in bands:

            # change intensity range randomly
            upper_bound = np.random.randint(1,128) 
            lower_bound = np.random.randint(0, upper_bound)

            # extract each band from the sample
            image = data_sample[band].copy()
            
            # apply to single bands as well as composed bands -> multi-channel image
            if image.ndim > 2:

                for i, channel in enumerate(image[:]):
                    # find minima and maxima per channel
                    min_ch = np.min(channel)
                    max_ch = np.max(channel)
                    # apply linear grey-value stretching
                    image[i] = np.clip((((channel-min_ch)*(upper_bound-lower_bound)/(max_ch-min_ch)) + lower_bound).astype(np.int8), 0,255)

            # just a single channel
            else:
                min_img = np.min(image)
                max_img = np.max(image)
                # apply linear grey-value stretching
                image = np.clip(((image-min_img)*(upper_bound-lower_bound)/(max_img-min_img)+lower_bound).astype(np.int8),0,255)

            # insert back into data_sample dictionary
            data_sample[band] = image

    return data_sample

def rnd_zoom(data_sample, probability=params.PROB):

    # extract images from data sample
    bands = [params.BAND, 'label']

    # arbitrary number in [0,1]
    rndm = np.random.rand()
    # arbitrary zoom factor
    f_zoom = np.random.uniform(1.0, 5.0)

    # apply only to defined percentage of images
    if rndm <= probability:
        for band in bands:
            image = data_sample[band].copy()

            # get dimensions
            img_height = image.shape[-2]
            img_width = image.shape[-1]

            # apply to single bands as well as composed bands -> multi-channel image
            if image.ndim > 2:

                for i, channel in enumerate(image[:]):
                    # insert back 
                    channel = cv2.resize(channel, None, fx=f_zoom, fy=f_zoom, interpolation= cv2.INTER_CUBIC)

                    # crop image back to original size
                    image[i] = channel[:img_height,:img_width]

            # just a single channel
            else:
                image = cv2.resize(image, None, fx=f_zoom, fy=f_zoom, interpolation= cv2.INTER_CUBIC)
                
                # crop image back to original size
                image = image[:img_height,:img_width]

            # insert back into data_sample dictionary
            data_sample[band] = image

    return data_sample

def h_flip(data_sample, probability=params.PROB):

    # extract images from data sample
    bands = [params.BAND, 'label']

    # arbitrary number in [0,1]
    rndm = np.random.rand()

    # apply only to defined percentage of images
    if rndm <= probability:
        for band in bands:
            image = data_sample[band].copy()

            # get dimensions
            img_height = image.shape[-2]
            img_width = image.shape[-1]

            # apply to single bands as well as composed bands -> multi-channel image
            if image.ndim > 2:

                for i, channel in enumerate(image[:]):
                    # insert back 
                    channel = cv2.flip(channel, 1)

            # just a single channel
            else:
                image = cv2.flip(image, 1)

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
