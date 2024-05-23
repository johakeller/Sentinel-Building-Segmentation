import torch
from torch.utils.data import Dataset
import os
import pickle
import random

from params import *


class SegmentationDataset(Dataset):

    def __init__(self, city, mode, used_patches=[]):
        # to check whether patch was already in the training set, to avoid data leakage
        self.used_patches = used_patches

        if mode == 'training':
            self.dataset = self.load_data(city, mode, dataset_size=TRAIN_SIZE)
        elif mode == 'validation':
            self.dataset = self.load_data(city, mode, dataset_size=VAL_SIZE)
        elif mode == 'test':
            self.dataset = self.load_test_data(city)

    # TODO dataloader necessary?
    def __len__(self):
        # return number of different images in one tensor (not bands)
        return self.dataset["RGB"].shape[0]
   
    # TODO dataloader necessary?
    def __getitem__(self, item):
        # extract one image (multiple bands) from the dataset dictionary
        rgb_out = self.dataset['RGB'][item]
        nirgb_out = self.dataset['NIRGB'][item]
        r_out = self.dataset['R'][item]
        g_out = self.dataset['G'][item]
        b_out = self.dataset['B'][item]
        nir_out = self.dataset['NIR'][item]
        label_out = self.dataset['label'][item]

        return {"RGB": rgb_out, "NIRGB": nirgb_out, "R":r_out, "G": g_out, 'B': b_out, 'NIR': nir_out, 'label': label_out}

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

    # helper function to load tensor for training or validation for a specific city
    def load_data(self, city, mode, path=IMAGE_DATA_PATH, patch_size=128, dataset_size=1280):

        # open pickle data
        with open(os.path.join(path, f'{city}.pkl'), 'rb') as fp:
            # TODO check if data is there
            training_data = pickle.load(fp)

        # get satellite image boundaries
        image_height = training_data['RGB'].shape[1]-1
        image_width = training_data['RGB'].shape[2]-1

        # output tensors for each band, dimensions: (N,C,H,W)
        rgb_out = torch.zeros(dataset_size, 3, patch_size, patch_size)
        nirgb_out = torch.zeros(dataset_size, 3, patch_size, patch_size)
        r_out = torch.zeros(dataset_size, 1, patch_size, patch_size)
        g_out = torch.zeros(dataset_size, 1, patch_size, patch_size)
        b_out = torch.zeros(dataset_size, 1, patch_size, patch_size)
        nir_out = torch.zeros(dataset_size, 1, patch_size, patch_size)
        label_out = torch.zeros(dataset_size, patch_size, patch_size)

        
        # extract dataset_size random patches
        for i in range(dataset_size):

            clouds = True # cloud cover 
            # defines if patch i was already used
            patch_used = True 

            while clouds or patch_used:
                # choose a patch of size patch_size at random coordinates (don't cross border)
                rnd_y = random.randint(0, image_height - patch_size)
                rnd_x = random.randint(0, image_width - patch_size)

                # image coordinates of the patch
                patch_coord = (rnd_y, rnd_x)

                # extract the same patch from all channels
                rgb_ch = training_data['RGB'][:,patch_coord[0]:patch_coord[0]+patch_size, patch_coord[1]:patch_coord[1]+patch_size]
                nirgb_ch = training_data['NIRGB'][:,patch_coord[0]:patch_coord[0]+patch_size, patch_coord[1]:patch_coord[1]+patch_size]
                r_ch = training_data['R'][:,patch_coord[0]:patch_coord[0]+patch_size, patch_coord[1]:patch_coord[1]+patch_size]
                g_ch = training_data['G'][:,patch_coord[0]:patch_coord[0]+patch_size, patch_coord[1]:patch_coord[1]+patch_size]
                b_ch = training_data['B'][:,patch_coord[0]:patch_coord[0]+patch_size, patch_coord[1]:patch_coord[1]+patch_size]
                nir_ch = training_data['NIR'][:,patch_coord[0]:patch_coord[0]+patch_size, patch_coord[1]:patch_coord[1]+patch_size]
                buildings_ch = training_data['Buildings'][patch_coord[0]:patch_coord[0]+patch_size, patch_coord[1]:patch_coord[1]+patch_size]
               
                # for training mode: if no clouds prepare data for dictionary
                if mode == 'training':
                    if not self.cloud_check(nir_ch):
                        # fill in output tensors
                        rgb_out[i] = torch.from_numpy(rgb_ch)
                        nirgb_out[i] = torch.from_numpy(nirgb_ch)
                        r_out[i] = torch.from_numpy(r_ch)
                        g_out[i] = torch.from_numpy(g_ch)
                        b_out[i] = torch.from_numpy(b_ch)
                        nir_out[i] = torch.from_numpy(nir_ch)
                        label_out[i] = torch.from_numpy(buildings_ch)
                        # set paramaters
                        self.used_patches.append(patch_coord) # append coordinates of patch to used patches list
                        clouds = False
                        patch_used = False
                # validation mode: also check if patch was not used in training
                else:
                    if not self.cloud_check(nir_ch) and not self.patch_check(patch_coord):
                        # fill in output tensors
                        rgb_out[i] = torch.from_numpy(rgb_ch)
                        nirgb_out[i] = torch.from_numpy(nirgb_ch)
                        r_out[i] = torch.from_numpy(r_ch)
                        g_out[i] = torch.from_numpy(g_ch)
                        b_out[i] = torch.from_numpy(b_ch)
                        nir_out[i] = torch.from_numpy(nir_ch)
                        label_out[i] = torch.from_numpy(buildings_ch)
                        clouds = False
                        patch_used = False

        # return dictionary containing all the tensors for each band
        return {"RGB": rgb_out, "NIRGB": nirgb_out, "R":r_out, "G": g_out, 'B': b_out, 'NIR': nir_out, 'label': label_out}

    def load_test_data(self, city, path=IMAGE_DATA_PATH):
        # open pickle data
        with open(os.path.join(path, f'{city}.pkl'), 'rb') as fp:
            # TODO check if data is there
            dataset = pickle.load(fp)
        # add dimension at 0
        return {"RGB": torch.from_numpy(dataset['RGB']).unsqueeze(0), "NIRGB": torch.from_numpy(dataset['NIRGB']).unsqueeze(0), "R":torch.from_numpy(dataset['R']).unsqueeze(0), "G": torch.from_numpy(dataset['G']).unsqueeze(0), 'B': torch.from_numpy(dataset['B']).unsqueeze(0), 'NIR': torch.from_numpy(dataset['NIR']).unsqueeze(0), 'label': torch.from_numpy(dataset['Buildings']).unsqueeze(0)}


