import numpy as np
import cv2

import params

def add_gaussian_noise(data_sample, probability=params.PROB, mean=params.GMEAN, sigma = params.STDDEV):
    '''
    Augmentation that adds Gaussian noise to all channels of an image.
    '''
    # arbitrary number in [0,1]
    rndm = np.random.rand()

    # apply only to defined percentage of images
    if rndm <= probability:
        # extract band from the sample
        image = data_sample[params.BAND].copy()

        # apply to single bands as well as composed bands -> multi-channel image
        if image.ndim > 2:
            for i, channel in enumerate(image[:]):
                # add Gaussian noise
                channel += np.random.normal(mean, sigma, channel.shape)

                # clip to normal intensity range
                channel = np.clip(channel,0,255)

        # just single channel
        else:
            # add Gaussian noise
            image += np.random.normal(mean, sigma, image.shape)
            
            # clip to normal intensity range
            image = np.clip(image,0,255)
            
        # insert back into data_sample dictionary
        data_sample[params.BAND] = image

    return data_sample

def salt_pepper_noise(data_sample, probability=params.PROB, sp_prob=params.SP_PROB):
    '''
    Augmentation that adds salt and pepper noise to all channels of an image.
    '''
    # arbitrary number in [0,1]
    rndm = np.random.rand()

    # apply only to defined percentage of images
    if rndm <= probability:
        # extract each band from the sample
        image = data_sample[params.BAND].copy()
        
        # get dimensions
        img_height = image.shape[-2]
        img_width = image.shape[-1]

        noise_no = int(image.size * sp_prob/2)

        # apply to single bands as well as composed bands -> multi-channel image
        if image.ndim > 2:
            for i, channel in enumerate(image[:]):
                # get coordinates where to apply noise
                coord_list = [[np.random.randint(0, img_height-1), np.random.randint(0, img_width-1)] for i in range(noise_no)]

                for pos in coord_list:
                    # arbitrary number in [0,1]
                    rndm_2 = np.random.rand()
                    if rndm_2 > 0.5:
                        # set white
                        channel[pos] = 1
                    else:
                        # set black
                        channel[pos] = 0

        # just a single channelS
        else:
            # get coordinates where to apply noise
            coord_list = [[np.random.randint(0, img_height-1), np.random.randint(0, img_width-1)] for i in range(noise_no)]

            for pos in coord_list:
                # arbitrary number in [0,1]
                rndm_3 = np.random.rand()
                if rndm_3 > 0.5:
                    # set white
                    image[pos] = 1
                else:
                    # set black
                    image[pos] = 0

        # insert back into data_sample dictionary
        data_sample[params.BAND] = image

    return data_sample

def rnd_zoom(data_sample, probability=params.PROB):
    '''
    Augmentation that applies a zoom by a random factor.
    '''
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

def apply_augment(data_sample, augmentations, probability=params.PROB):
    '''
    Create pipeline of augmentation functions and apply them to the input
    '''

    output = data_sample

    # apply only to defined percentage of images
    if np.random.rand() <= probability:
        # check if a list of augmentations is passed
        if isinstance(augmentations, list):
            for augmentation in augmentations:
                output = augmentation(output, probability=1.0)
            return output
    else:
        # only single augmentation is passed
        return augmentations(output)
    
    # no augmentation applied
    return output