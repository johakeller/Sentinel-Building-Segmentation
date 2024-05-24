from params import *
from data_acquisition import *
from dataset import *
from train_apply import *

# what dataset is loaded, how many bands, how to train etc.?
class DataSplit():
    def __init__(self, city_list=CITIES):
        self.training_sets = []
        self.validation_sets = []
        # inititalize training and validation data for list of given cities
        for city in city_list:
            self.training_sets.append(SegmentationDataset(city, 'training'))
            self.training_sets.append(SegmentationDataset(city, 'validation'))

        self.test_set = SegmentationDataset(TEST_CITY, 'test')



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
