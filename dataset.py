import torch
from torch.utils.data import Dataset
import os
import pickle
import random


from params import *

def load_training_data(path=IMAGE_DATA_PATH, patch_size=64, city='Berlin', dataset_size=128):

    # open pickle data
    with open(os.path.join(path, f'{city}.pkl'), 'rb') as fp:
        # TODO check if data is there
        training_data = pickle.load(fp)

    # get satellite image boundaries
    image_height = training_data['RGB'].shape[1]-1
    image_width = training_data['RGB'].shape[2]-1

    # output tensors for each band, dimensions: (N,C, H, W)
    rgb_out = torch.zeros(dataset_size, 3, patch_size, patch_size)
    nirgb_out = torch.zeros(dataset_size, 3, patch_size, patch_size)
    r_out = torch.zeros(dataset_size, patch_size, patch_size)
    g_out = torch.zeros(dataset_size, patch_size, patch_size)
    b_out = torch.zeros(dataset_size, patch_size, patch_size)
    nir_out = torch.zeros(dataset_size, patch_size, patch_size)
    label_out = torch.zeros(dataset_size, patch_size, patch_size)

    
    # extract dataset_size random patches
    for i in range(dataset_size):
        # choose a patch of size patch_size at random coordinates (don't cross border)
        rnd_y = random.randint(0, image_height - patch_size)
        rnd_x = random.randint(0, image_width - patch_size)

        # extract the same patch from all channels
        rgb_ch = training_data['RGB'][:,rnd_y:rnd_y+patch_size, rnd_x:rnd_x+patch_size]
        # print(rgb_ch.shape)
        nirgb_ch = training_data['NIRGB'][:,rnd_y:rnd_y+patch_size, rnd_x:rnd_x+patch_size]
        r_ch = training_data['R'][:,rnd_y:rnd_y+patch_size, rnd_x:rnd_x+patch_size]
        g_ch = training_data['G'][:,rnd_y:rnd_y+patch_size, rnd_x:rnd_x+patch_size]
        b_ch = training_data['B'][:,rnd_y:rnd_y+patch_size, rnd_x:rnd_x+patch_size]
        nir_ch = training_data['NIR'][:,rnd_y:rnd_y+patch_size, rnd_x:rnd_x+patch_size]
        buildings_ch = training_data['Buildings'][rnd_y:rnd_y+patch_size, rnd_x:rnd_x+patch_size]

        # TODO apply cloud classifier before inserting

        # fill in output tensors
        rgb_out[i] = torch.from_numpy(rgb_ch)
        nirgb_out[i] = torch.from_numpy(nirgb_ch)
        r_out[i] = torch.from_numpy(r_ch)
        g_out[i] = torch.from_numpy(g_ch)
        b_out[i] = torch.from_numpy(b_ch)
        nir_out[i] = torch.from_numpy(nir_ch)
        label_out[i] = torch.from_numpy(buildings_ch)

    # return dictionary containing all the tensors for each band
    return {"RGB": rgb_out, "NIRGB": nirgb_out, "R":r_out, "G": g_out, 'B': b_out, 'NIR': nir_out, 'label': label_out}

class SegmentationDataset(Dataset):

    def __init__(self):
        self.dataset = load_training_data();

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


training_data = load_training_data()
#print(training_data)