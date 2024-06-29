import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import params

class Trainer:

    def __init__(self, model, train_loader=None, val_loader=None, test_loader=None, mode = 'train', band = params.BAND, epochs=params.EPOCHS, lr = params.LEARNING_RATE, train_output=None, val_output=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mode = mode
        self.epochs = epochs
        self.band = band
        self.model = model # initialize model after bands, because number of channels has to be adjusted
        self.train_output = train_output
        self.val_output = val_output
        self.optimizer = optim.Adam(model.parameters(), lr =lr)
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
    
    def training(self):
        '''
        Start training.
        '''
        # set model to train mode
        self.model.train()
        for epoch in range(self.epochs): # all epochs

            all_labels = torch.tensor([])
            all_predictions = torch.tensor([])

            for city, dataloader in self.train_loader.items(): # go through the dictionary train loader (for each city)
                # statistics 
                prog_bar = tqdm(total=len(dataloader.dataset), desc=f'{city} training ', position=0, leave=False)
                avg_loss = 0.0

                for data in dataloader: # go through data in each data loader
                    # update prog_bar
                    prog_bar.update(dataloader.batch_size)

                    train_input = data[self.band]
                    train_label = data['label']

                    prediction = self.model.forward(train_input) # forward pass
                    loss = self.criterion(prediction, train_label) # calculate error
                    avg_loss += loss.item()

                    # backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # TODO: look up other metrics, relevant for image segmentation AUTOMATIC IMAGE ANALYSIS
                    # for metrics (remove unnecessary first dimension)
                    all_labels = torch.cat((all_labels, train_label.flatten().detach()))
                    all_predictions = torch.cat((all_predictions, (prediction > params.PRED_THRESHOLD).int().flatten().detach()))
                
                prog_bar.close()             
                # average loss after each city
                message = f'Epoch {epoch + 1}, {city}: avg loss: {round(avg_loss/ len(dataloader),2)}\n'
                print(message)
                
            # calculate, print and write metrics
            calculate_metrics(all_labels, all_predictions, self.train_output)

    def validation(self):
        '''
        Start testing.
        '''
        # set model to validation mode
        self.model.eval()

        avg_loss = 0.0
        all_labels = torch.tensor([])
        all_predictions = torch.tensor([])

        for city, dataloader in self.train_loader.items(): # go through the dictionary train loader (for each city)

            # statistics 
            prog_bar = tqdm(total=len(dataloader.dataset), desc=f'{city} validation', position=0, leave=True)
            inp = None
            pred = None
            lab = None

            with torch.no_grad():
                for data in dataloader: # go through data in each data loader
                    # update progress prog_bar
                    prog_bar.update(dataloader.batch_size)

                    test_input = data[self.band]
                    test_label = data['label']
                    #print(f'label: {test_label.shape}, type {type(test_label)}')

                    prediction = self.model.forward(test_input) # forward pass
                    loss = self.criterion(prediction, test_label) # calculate error
                    avg_loss += loss.item()

                    # for visualization
                    inp = test_input[0]
                    pred = prediction[0]
                    lab = test_label[0]

                    # TODO: look up other metrics, relevant for image segmentation AUTOMATIC IMAGE ANALYSIS
                    # for metrics (remove unnecessary first dimension)
                    all_labels = torch.cat((all_labels, test_label.flatten().detach()))
                    all_predictions = torch.cat((all_predictions, (prediction > params.PRED_THRESHOLD).int().flatten().detach()))

            prog_bar.close()
            # average loss per city
            message = f'Valdiation {city}: avg loss: {round(avg_loss/ len(dataloader),2)}\n'
            print(message)

            # DELETE
            visualize_test(inp, lab, pred, 'input', 'label', 'prediction')

        # calculate, print and write metrics
        calculate_metrics(all_labels, all_predictions, self.val_output)

def visualize_test(inp, lab, pred, inp_title, lab_title, pred_title):
    # Create a new figure for the subplots
    plt.figure(figsize=(15, 5))

    # List of images and titles
    images = [inp, lab, pred]
    titles = [inp_title, lab_title, pred_title]

    for i in range(3):
        # Get the tensor from the 'R' channel, remove the first dimension (1, 128, 128) -> (128, 128)
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

def calculate_metrics(labels, predictions, output):
    '''
    Function that computes based on the probabilities for each class (processed model output)
    the average confidence per label, the value of the lowest confidence across all labels and 
    the row index and column index of this sample with the lowest confidence.

    Args:
        probability (numpy.ndarray): array of softmaxed model output (probabilities for each class)

    Returns:
        tuple: (avg_conf, lowest_conf, id_r, id_c)
            avg_conf (float): average confidence (mean of the entire matrix)
            lowest_conf (float): lowest confidence value in the entire matrix
            id_r (int): row index of the lowest confidence value
            id_c (int): column index of the lowest confidence value
    '''

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

    # print and save metrics
    message = f'Default ConvNet: accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1 score: {f1:.2f}\n'
    print(message)
    write_results(message + '\n', output)

def write_results(output, file):
    '''
    Helper function that writes output into a .txt file in the output_folder.

    Args:
        output (str): message
        file (str): file path

    Returns:
        None
    '''

    os.makedirs(params.OUT_PATH, exist_ok=True)

    with open(os.path.join(params.OUT_PATH, f'{file}.txt'), "a", encoding="utf-8") as file:
        file.write(output)

