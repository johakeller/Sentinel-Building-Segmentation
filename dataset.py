'''Module to create a dataset for the satellite image data which was preprocessed in 
data_acquisition. '''

import os
import random
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

import params
import augment


class SegmentationDataset(Dataset):
    '''
    Class to create a dataset out of satellite image bands for a given city, constructs image patches 
    with corresponding building labels. The options training, validation and test are available. The 
    data is saved and loaded dynamically, using a memmap format, in the dataloader.
    
    Args:
        city (str): name of the city 
        mode (str): type of dataset: train, validation or test
        patch_size (int): size of square patches to extract for training
        building_cover: percentage of patch covered with buildings
        used_patches (list): for validation set: must know which patches have been used in the training set already
        dataset_size (int): number of patches 
        train_path (str): path for training dataset
        val_path (str): path for validation dataset
        test_path (str): path for test dataset
        augmentations (callable): augmentations to be applied to the training data

    Attributes:
        city (str): name of the city 
        mode (str): type of dataset: train, validation or test
        patch_size (int): size of square patches to extract for training
        building_cover: percentage of patch covered with buildings
        used_patches (list): for validation set: must know which patches have been used in the training set already
        dataset_size (int): number of patches 
        ch_num (int): number of channels in dataset
        augmentations (callable): augmentations to be applied to the training data
        dataset_path (str): path to dataset
        data_file_path (str): path to memmap data file
        label_path (str): path to memmap label file

    Methods:
        __len__(): Returns length of the dataset
        __getitem__(item): returns data sample
        cloud_check (patch, contr_thresh, bright_thresh): checks if amount of cloud cover is below a certain threshold
        patch_check (patch_coord): checks for validation set, if patch was already used in training set
        building_check (patch): checks if percentage of buildings is above a certain threshold in the patch
        dynamic_save (dataset): savs the data as numpy memory map for lazy loading
        create_dataset(): creates an entire dataset with train, validation and test section
    '''

    def __init__(self, city, mode, patch_size = params.PATCH_SIZE, building_cover = params.BUILDING_COVER, used_patches=None, dataset_size=None, train_path=params.DATASET_TRAIN, val_path=params.DATASET_VAL, test_path=params.DATASET_TEST, augmentations = None):

        if used_patches is None:
            used_patches = []
        self.used_patches = used_patches # to check whether patch was already in the training set, to avoid data leakage
        self.city = city
        self.mode = mode
        self.patch_size=patch_size
        self.building_cover =building_cover
        self.dataset_size = dataset_size
        self.ch_num = 4
        self.augmentations = augmentations # apply augmentation to the dataset

        # define type of the dataset
        if self.mode =='training':
            self.dataset_path = train_path
        elif self.mode == 'validation':
            self.dataset_path = val_path
        elif self.mode == 'test':
            self.dataset_path = test_path

        # if the dataset already exists, don't recreate it
        self.data_file_path = os.path.join(self.dataset_path, f'{self.city}_data.mmap')
        self.label_path = os.path.join(self.dataset_path, f'{self.city}_label.mmap')

        # create the path dictionary based on existing files
        if os.path.exists(os.path.join(self.dataset_path, f'{self.city}_data.mmap')): 
            self.dataset = {'data': [self.data_file_path, (self.dataset_size, self.ch_num, self.patch_size, self.patch_size)], 'label': [self.label_path, (self.dataset_size, 1, self.patch_size, self.patch_size)] }
        
        # if the dataset does not exist, create it
        else:
            self.dataset = self.create_dataset()

    def __len__(self):
        '''
        Obtain the length of the entire data tensor (band-wise).

        Returns:
            int: length of the data tensor
        '''

        # return number of different images in one tensor (not bands)
        return self.dataset_size
   
    def __getitem__(self, item):
        '''
        Obtain single data sample containing all bands dynamically from the saved tensor. Applies
        an augmentation or an augmentatin pipeline to the sample with a certain probability if 
        these parameters are set.

        Args:
            item (int): index of the desired data sample in the data tensor
        
        Returns:
            dict: dictionary with pytorch tensor for each band
        '''

        # extract multiple bands from the memmap dynamically
        data = np.memmap(self.dataset['data'][0],'float32', mode='r+', shape=self.dataset['data'][1])

        # extract single channels from data
        r_out = np.expand_dims(data[item][0], axis=0)
        g_out = np.expand_dims(data[item][1], axis=0)
        b_out = np.expand_dims(data[item][2], axis=0)
        nir_out = np.expand_dims(data[item][3], axis=0)

        # and from label
        label_data = np.memmap(self.dataset['label'][0],'float32', mode='r+', shape=self.dataset['label'][1])
        label_out = label_data[item]
        
        # create all, rgb and nirgb from single channels
        rgb_out = np.stack((data[item][0], data[item][1], data[item][2]),axis=0)
        nirgb_out = np.stack((data[item][3], data[item][1], data[item][2]),axis=0)
        all_out = np.stack((data[item][3],data[item][0], data[item][1], data[item][2]),axis=0)

        # output of the getitem method: dictionary with pytorch tensor per band
        data_sample = {"all": all_out,"RGB": rgb_out, "NIRGB": nirgb_out, "R": r_out, "G": g_out, 'B': b_out, 'NIR': nir_out, 'label': label_out}

        # apply augmentation if parameter set
        if self.augmentations is not None:
            data_sample = augment.apply_augment(data_sample, self.augmentations)

        # convert into torch tensor
        for key, value in data_sample.items():
            # copy values and move to device
            data_sample[key] = torch.from_numpy(np.copy(value)).to(params.DEVICE)
       
        return data_sample

    def cloud_check(self, patch, contr_thresh=0.8, bright_thresh=0.8):
        '''
        Simple cloud cover classifier: Check the image patch for cloud cover and returns False if the cloud 
        cover is above the threshold parameters.

        Args:
            patch (np.array): satellite image patch in NIR 
            contr_threshold (float): contrast threshold
            bright_thresh (float): brighntess threshold
        
        Returns:
            bool: Is image patch cloud covered according to method?
        '''

        # contrast is variance
        contrast = patch.var().item()
        # use also brightness since clouds appear bright in NIR
        brightness = patch.mean().item()
        if contrast < contr_thresh and brightness > bright_thresh:
            return False
        else:
            return True
    
    def patch_check(self, patch_coord):
        '''
        Check if image patch is already (partly) contained in the training set to avoid data leakage. 
        Returns False if the case.

        Args:
            patch_coord (list): corner points of the patch in image coordinates

        Returns:
            bool: Patch either falls within the coordinates of a patch used in the training set or not. 
        '''
        patch_corners = [(patch_coord[0],patch_coord[1]), (patch_coord[0]+64,patch_coord[1]), (patch_coord[0],patch_coord[1]+64), (patch_coord[0]+64,patch_coord[1]+64)] # [(y,x), (y+64, x), (y, x+64), (y+64, x+64)]
        
        # check if patch_corners fall inside on of the used patches
        for used_patch in self.used_patches:
            for point in patch_corners:
                if ((point[0] >= used_patch[0]) and (point[0] <= used_patch[0]+64)) and ((point[1] >= used_patch[1]) and (point[1] <= used_patch[1]+64)):
                    return False

        return True
    
    def building_check(self, patch):
        '''
        Check if the percentage of the patch covered with buildings is below a threshold. 
        Returns False in that case. Obtain a desired label distribution for the training 
        set. 

        Args:
            patch (np.array): binary building label tensor
        
        Returns:
            bool: True if the building cover percentage larger equal the threshold
        '''

        avg_build = patch.mean().item()
        if avg_build < self.building_cover: 
            return False
        return True
    
    def dynamic_save(self, dataset):
        '''
        Helper function for dynamic saving of the output tensor via numpy memorymap. Allows 
        to access indices without loading the entire tensor into memory.

        Args:
            dataset (dict): dataset containing tensors of all the bands for a city
        
        Returns:
            string: path of the memorymap tensor 
        '''
        # create dataset path if not there
        if not os.path.isdir(self.dataset_path):
            os.makedirs(self.dataset_path)

        # init output path dictionaries for the tensor
        dataset_path = {'data': None, 'label': None}

        for key in dataset.keys():
            file_path=os.path.join(self.dataset_path, f'{self.city}_{key}.mmap')
            dataset_map = np.memmap(file_path, dtype='float32', mode='w+', shape=dataset[key].shape)
            dataset_path[key]=[file_path, dataset[key].shape] 
            # write to memmap
            dataset_map[:] = dataset[key][:] 
            # flush to disc
            dataset_map.flush() 

        print(f'Dataset {self.city} {self.mode} written.')
        return dataset_path
        
    def create_dataset(self):
        '''
        Initiate a new dataset given the satellite imagery for a city. Extracts randomly 
        a predefined number of patches of predefined size from all bands in parallel and 
        saves them dynamically to disc in a tensor with dimensions: (N,C,H,W). Patches 
        are checked to have no cloud cover, sufficient builings, and for the vailidation 
        set to be not present in the training set. The corresponding parameters can be 
        found in params.py.

        Raises:
            FileNotFoundError: Satellite image data for the city is not found in the given path

        Returns:
            dict: path of the memory maps of the data tensor
        '''

        images_path = os.path.join(params.IMAGE_DATA_PATH, f'{self.city}.pkl')

        # check if pkl data exists
        if not os.path.exists(images_path):
            raise FileNotFoundError(f'The image data for {self.city} is not found.')
        
        # open pickle data
        with open(images_path, 'rb') as fp:
            raw_data = pickle.load(fp)

            # get satellite image boundaries
            image_height = raw_data['R'].shape[0]-1
            image_width = raw_data['R'].shape[1]-1

            # output tensors for each band, dimensions: (N,C,H,W)
            bands_out = np.zeros((self.dataset_size, 4, self.patch_size, self.patch_size))
            labels_out = np.zeros((self.dataset_size, 1, self.patch_size, self.patch_size))

            # extract dataset_size random patches (i iterates through N of (N,C,H,W))
            for i in range(self.dataset_size):
                
                # while there has not been a proper candidate for patch i
                keep_looking = True

                # keep looking for new patches, while the patch was already used or too many clouds
                while keep_looking:
                    # choose a patch of size patch_size at random coordinates (don't cross border)
                    rnd_y = random.randint(0, image_height - self.patch_size)
                    rnd_x = random.randint(0, image_width - self.patch_size)

                    # image coordinates of the patch
                    patch_coord = (rnd_y, rnd_x)

                    # extract the same patch from all channels
                    r_ch = raw_data['R'][patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    g_ch = raw_data['G'][patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    b_ch = raw_data['B'][patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    nir_ch = raw_data['NIR'][patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    buildings_ch = raw_data['Buildings'][patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                
                    # for training mode: if no clouds prepare data for dictionary
                    if self.mode == 'training':
                        if self.cloud_check(nir_ch) and self.building_check(buildings_ch):
                            # fill in output tensors: bands_out dim = (N,C,H,W)
                            bands_out[i][0] = r_ch
                            bands_out[i][1] = g_ch
                            bands_out[i][2] = b_ch
                            bands_out[i][3] = nir_ch
                            labels_out[i][0] = buildings_ch
                            # set paramaters
                            self.used_patches.append(patch_coord) # append coordinates of patch to used patches list
                            # set loop parameter
                            keep_looking = False

                    # validation mode: check for cloud cover, also check if patch was not used in training, also check for min. building cover
                    else:
                        if self.cloud_check(nir_ch) and self.patch_check(patch_coord) and self.building_check(buildings_ch):
                            # fill in output tensors: bands_out dim = (N,C,H,W)
                            bands_out[i][0] = r_ch
                            bands_out[i][1] = g_ch
                            bands_out[i][2] = b_ch
                            bands_out[i][3] = nir_ch
                            labels_out[i][0] = buildings_ch
                            keep_looking = False
            dataset = {"data": bands_out, 'label': labels_out}
            dataset_path = self.dynamic_save(dataset)

        return dataset_path

        