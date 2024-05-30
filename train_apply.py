from dataset import SegmentationDataset
from torch.utils.data import DataLoader

from models import *
from params import *
from data_acquisition import *
from train_apply import *

# what dataset is loaded, how many bands, how to train etc.?
class DataSplit():
    def __init__(self, city_list=CITIES, train_size = TRAIN_SIZE, val_size = VAL_SIZE, test_size=TEST_SIZE, batch_size=BATCH_SIZE, patch_size = PATCH_SIZE, building_cover = BUILDING_COVER, augmentation=None):
        # create Dataloaders in dictinary for given lists of cities
        self.train_loader = {city:DataLoader(SegmentationDataset(city, 'training', dataset_size=train_size), batch_size) for city in city_list} # dictionary of dataloaders
        self.val_loader= {city:DataLoader(SegmentationDataset(city, 'validation', dataset_size=val_size), batch_size) for city in city_list} # dictionary of dataloaders
        self.test_loader = DataLoader(SegmentationDataset(TEST_CITY, 'test', dataset_size=test_size), batch_size) # only dataloader!
    


def train_apply(method=None):
    input_channels = 3
    dataset = DataSplit() # init dataloader for train, valdiation, test
    model = ConvNet(input_channels) # define input channels 




    ############################################################################### TEST #######################################################################################
    def visualize_test(self):
        #DELETE
        import matplotlib.pyplot as plt

        # Assuming validation_dataset is an instance of SegmentationDataset
        # and the 'R' channel is the one you want to plot

        # Get the tensor for the 'R' channel, remove the first dimension (1, 128, 128) -> (128, 128)
        r_channel_tensor = self.test_set.__getitem__(0)['NIR'].squeeze(0)

        # Convert the tensor to a NumPy array
        r_channel_array = r_channel_tensor.numpy()

        # Plot the greyscale image
        plt.imshow(r_channel_array, cmap='gray')
        plt.colorbar()  # Add a colorbar for reference
        plt.title(f"Greyscale Image from 'NIR' Channel\ncontrast = {r_channel_tensor.std().item()}\nbrightness = {r_channel_tensor.mean().item()}")
        plt.show()
