import torch
from torch.utils.data import Dataset
import os
import pickle
import random
import numpy as np

from params import *


class SegmentationDataset(Dataset):

    def __init__(self, city, mode, used_patches=[]):
        # to check whether patch was already in the training set, to avoid data leakage
        self.used_patches = used_patches
        self.city = city
        self.mode = mode
        self.patch_size=PATCH_SIZE
        if self.mode =='training':
            self.dataset_size = TRAIN_SIZE
            self.dataset = self.create_dataset(self.city)
        elif self.mode == 'validation':
            self.dataset_size = VAL_SIZE
            self.dataset = self.create_dataset(self.city)
        elif self.mode == 'test':
            self.dataset_size = 1
            self.dataset = self.create_dataset(self.city)

    def __len__(self):
        # return number of different images in one tensor (not bands)
        return self.dataset_size
   
    def __getitem__(self, item):
        # extract one image (multiple bands) from the memmap dynamically
        rgb_out = np.memmap(self.dataset['RGB'][0],'float32', mode='r', shape=self.dataset['RGB'][1])
        nirgb_out = np.memmap(self.dataset['NIRGB'][0],'float32', mode='r', shape=self.dataset['NIRGB'][1])
        r_out = np.memmap(self.dataset['R'][0],'float32', mode='r', shape=self.dataset['R'][1])
        g_out = np.memmap(self.dataset['G'][0],'float32', mode='r', shape=self.dataset['G'][1])
        b_out = np.memmap(self.dataset['B'][0],'float32', mode='r', shape=self.dataset['B'][1])
        nir_out = np.memmap(self.dataset['NIR'][0],'float32', mode='r', shape=self.dataset['NIR'][1])
        label_out = np.memmap(self.dataset['label'][0],'float32', mode='r', shape=self.dataset['label'][1])

        # output of the getitem method: dictionary with pytorch tensor per band
        return {"RGB": torch.from_numpy(rgb_out[item]), "NIRGB": torch.from_numpy(nirgb_out[item]), "R":torch.from_numpy(r_out[item]), "G": torch.from_numpy(g_out[item]), 'B': torch.from_numpy(b_out[item]), 'NIR': torch.from_numpy(nir_out[item]), 'label': torch.from_numpy(label_out[item])}

    def cloud_check(self, patch, contr_thresh=0.8, bright_thresh=0.8):
        # simple cloud classfier by checking RMS contrast: https://en.wikipedia.org/wiki/Contrast_(vision)#RMS_contrast
        # contrast is standard deviation
        contrast = patch.std().item()
        # use also brightness since clouds appear bright in NIR
        brightness = patch.mean().item()
        if contrast < contr_thresh and brightness > bright_thresh:
            return True
        else:
            #print(f'contrast: {contrast}')
            #print(f'brightness: {brightness}')
            return False
    
    def patch_check(self, patch_coord):
        patch_corners = [(patch_coord[0],patch_coord[1]), (patch_coord[0]+64,patch_coord[1]), (patch_coord[0],patch_coord[1]+64), (patch_coord[0]+64,patch_coord[1]+64)] # [(y,x), (y+64, x), (y, x+64), (y+64, x+64)]
        
        # check if patch_corners fall inside on of the used patches
        for used_patch in self.used_patches:
            for point in patch_corners:
                if ((point[0] >= used_patch[0]) and (point[0] <= used_patch[0]+64)) and ((point[1] >= used_patch[1]) and (point[1] <= used_patch[1]+64)):
                    return True

        return False

    def dynamic_save(self, dataset):
        # create dataset path if not there
        if not os.path.isdir(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        # init output path dictionaries for the tensor
        dataset_path = {"RGB": None, "NIRGB": None, "R":None, "G": None, 'B': None, 'NIR': None, 'label': None}

        for band in dataset.keys():
            file_path=os.path.join(DATASET_PATH, f'{self.city}{band}.mmap')
            dataset_map = np.memmap(file_path, dtype='float32', mode='w+', shape=dataset[band].shape)
            dataset_path[band]=[file_path, dataset[band].shape] 
            dataset_map[:] = dataset[band][:] # write to memmap
            dataset_map.flush() # flush to disc

        print(f'Dataset {self.city} {self.mode} written.')
        return dataset_path
        
    def create_dataset(self, path=IMAGE_DATA_PATH):
        images_path = os.path.join(path, f'{self.city}.pkl')

        # check if data exists
        if not os.path.exists(images_path):
            raise FileNotFoundError(f'The image data for {self.city} is not found.')
        
        # open pickle data
        with open(images_path, 'rb') as fp:
            raw_data = pickle.load(fp)

        if self.mode =='test':
            dataset = {"RGB": raw_data['RGB'][np.newaxis,:], "NIRGB": raw_data['NIRGB'][np.newaxis,:], "R":raw_data['R'][np.newaxis,:], "G": raw_data['G'][np.newaxis,:], 'B': raw_data['B'][np.newaxis,:], 'NIR': raw_data['NIR'][np.newaxis,:], 'label': raw_data['Buildings'][np.newaxis,:]}
            dataset_path = self.dynamic_save(dataset)
        else:
            # get satellite image boundaries
            image_height = raw_data['RGB'].shape[1]-1
            image_width = raw_data['RGB'].shape[2]-1

            # output tensors for each band, dimensions: (N,C,H,W)
            rgb_out = np.zeros((self.dataset_size, 3, self.patch_size, self.patch_size))
            nirgb_out = np.zeros((self.dataset_size, 3, self.patch_size, self.patch_size))
            r_out = np.zeros((self.dataset_size, 1, self.patch_size, self.patch_size))
            g_out = np.zeros((self.dataset_size, 1, self.patch_size, self.patch_size))
            b_out = np.zeros((self.dataset_size, 1, self.patch_size, self.patch_size))
            nir_out = np.zeros((self.dataset_size, 1, self.patch_size, self.patch_size))
            label_out = np.zeros((self.dataset_size, self.patch_size, self.patch_size))

            
            # extract dataset_size random patches
            for i in range(self.dataset_size):

                clouds = True # cloud cover 
                # defines if patch i was already used
                patch_used = True 

                while clouds or patch_used:
                    # choose a patch of size patch_size at random coordinates (don't cross border)
                    rnd_y = random.randint(0, image_height - self.patch_size)
                    rnd_x = random.randint(0, image_width - self.patch_size)

                    # image coordinates of the patch
                    patch_coord = (rnd_y, rnd_x)

                    # extract the same patch from all channels
                    rgb_ch = raw_data['RGB'][:,patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    nirgb_ch = raw_data['NIRGB'][:,patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    r_ch = raw_data['R'][:,patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    g_ch = raw_data['G'][:,patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    b_ch = raw_data['B'][:,patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    nir_ch = raw_data['NIR'][:,patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                    buildings_ch = raw_data['Buildings'][patch_coord[0]:patch_coord[0]+self.patch_size, patch_coord[1]:patch_coord[1]+self.patch_size]
                
                    # for training mode: if no clouds prepare data for dictionary
                    if self.mode == 'training':
                        if not self.cloud_check(nir_ch):
                            # fill in output tensors
                            rgb_out[i] = rgb_ch
                            nirgb_out[i] = nirgb_ch
                            r_out[i] = r_ch
                            g_out[i] = g_ch
                            b_out[i] = b_ch
                            nir_out[i] = nir_ch
                            label_out[i] = buildings_ch
                            # set paramaters
                            self.used_patches.append(patch_coord) # append coordinates of patch to used patches list
                            clouds = False
                            patch_used = False
                    # validation mode: also check if patch was not used in training
                    else:
                        if not self.cloud_check(nir_ch) and not self.patch_check(patch_coord):
                            # fill in output tensors
                            rgb_out[i] = rgb_ch
                            nirgb_out[i] = nirgb_ch
                            r_out[i] = r_ch
                            g_out[i] = g_ch
                            b_out[i] = b_ch
                            nir_out[i] = nir_ch
                            label_out[i] = buildings_ch
                            clouds = False
                            patch_used = False
            dataset = {"RGB": rgb_out, "NIRGB": nirgb_out, "R":r_out, "G": g_out, 'B': b_out, 'NIR': nir_out, 'label': label_out}
            dataset_path = self.dynamic_save(dataset)

        return dataset_path

    