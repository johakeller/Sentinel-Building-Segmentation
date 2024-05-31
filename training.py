import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from params import *

class Trainer:

    def __init__(self, model, train_loader=None, val_loader=None, test_loader=None, mode = 'train', band = 'RGB', epochs=EPOCHS, lr = LEARNING_RATE):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self. test_loader = test_loader
        self.mode = mode
        self.epochs = epochs
        self.band = band
        self.optimizer = optim.Adam(model.parameters(), lr =lr)
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
    
    def training(self):
        for epoch in range(self.epochs): # all epochs

            for city, dataloader in self.train_loader.items(): # go through the dictionary train loader (for each city)

                # statistics 
                self.bar = tqdm(total=len(dataloader.dataset), desc=f'{city} training ', position=0)
                avg_loss = 0.0
                inp = None
                pred = None
                lab = None


                for data in dataloader: # go through data in each data loader
                    # update progress bar
                    self.bar.update(dataloader.batch_size)

                    input = data[self.band]
                    labels = data['label']

                    predictions = self.model.forward(input) # forward pass
                    loss = self.criterion(predictions, labels) # calculate error
                    avg_loss += loss.item()

                    # backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    inp = input[0]
                    pred = predictions[0]
                    lab = labels[0]

                
                message = f'\nEpoch {epoch}: avg loss: {round(avg_loss/ len(dataloader),2)}\n'
                print(message)


                # DELETE
                visualize_test(inp, lab, pred, 'input', 'label', 'prediction')
                # reset progress bar --> REPLACE
                self.bar.n = 0
                self.bar.last_print_n = 0
                self.bar.refresh()
                    

    ############################################################################### TEST #######################################################################################

def visualize_test(inp, lab, pred, inp_title, lab_title, pred_title):
    # Create a new figure for the subplots
    plt.figure(figsize=(15, 5))

    # List of images and titles
    images = [inp, lab, pred]
    titles = [inp_title, lab_title, pred_title]

    for i in range(3):
        # Get the tensor for the 'R' channel, remove the first dimension (1, 128, 128) -> (128, 128)
        r_channel_tensor = images[i][0]

        # Convert the tensor to a NumPy array
        r_channel_array = r_channel_tensor.detach().numpy()

        # Plot the greyscale image in the specified subplot
        plt.subplot(1, 3, i+1)
        plt.imshow(r_channel_array, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')  # Hide axes for better visualization

    # Show the combined figure
    plt.show()

