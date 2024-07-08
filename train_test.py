import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import params
#TODO DELETE
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
        plt.axis('off')  # Hide axes 

    # Show the combined figure
    plt.show()

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

class Trainer:

    def __init__(self, model, train_loader=None, val_loader=None, test_loader=None, mode = 'train', epochs=params.EPOCHS, lr = None, train_output=None, val_output=None, band=None, dropout=None, weight_decay=None, model_name=None, class_weight=1.0, lr_scheduler=False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mode = mode
        self.epochs = epochs
        self.band = band
        self.description = f'{model_name}, bands: {band}, learning rate: {lr}, weight decay: {weight_decay}'
        self.model = model # initialize model after bands, because number of channels has to be adjusted
        self.train_output = train_output
        self.val_output = val_output
        self.optimizer = optim.Adam(model.parameters(), lr =lr, weight_decay=weight_decay)
        if lr_scheduler:
            self.lr_scheduler = StepLR(self.optimizer, step_size = 4, gamma = 0.5) 
        else:
            self.lr_scheduler = None
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=class_weight)

    
    def training(self):
        '''
        Start training.
        '''
        # print hyperparameters
        print(self.description)

        # set model to train mode
        self.model.train()
        for epoch in range(self.epochs): # all epochs
            all_labels = torch.Tensor([])
            all_predictions = torch.Tensor([])

            for city, dataloader in self.train_loader.items(): # go through the dictionary train loader (for each city)
                # statistics 
                prog_bar = tqdm(total=len(dataloader.dataset), desc=f'Training epoch {epoch+1}, {city}', position=0, leave=False)
                avg_loss = 0.0

                # TODO delete
                inp = None
                pred = None
                lab = None

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

                    # TODO delete 
                    inp = train_input[3]
                    pred = prediction[3]
                    lab = train_label[3]
                    
                    # TODO DELETE
                    #visualize_test(inp, lab, pred, 'input', 'label', 'prediction')

                    # for metrics (remove unnecessary first dimension)
                    all_labels = torch.cat((all_labels, train_label.flatten().detach()))
                    # create binary predictions out of probabilities by thresholding
                    all_predictions = torch.cat((all_predictions, (prediction > params.PRED_THRESHOLD).flatten().detach()))
                # learning rate scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                prog_bar.close()             
                # average loss after each city
                message = f'Training epoch {epoch + 1}, {city}, avg loss: {round(avg_loss/ len(dataloader),2)}\n'
                print(message)

                # TODO DELETE
                #visualize_test(inp, lab, pred, 'input', 'label', 'prediction')
 
        # calculate, print and write metrics, return f1 score
        return self.calculate_metrics(all_labels, all_predictions, self.train_output, mode='Training')


    def validation(self):
        '''
        Start validation.
        '''
        # set model to validation mode
        self.model.eval()

        avg_loss = 0.0
        all_labels = torch.tensor([])
        all_predictions = torch.tensor([])

        for city, dataloader in self.val_loader.items(): # go through the dictionary validation loader (for each city)

            # statistics 
            prog_bar = tqdm(total=len(dataloader.dataset), desc=f'{city} validation', position=0, leave=False)
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

                    # DELETE
                    #visualize_test(inp, lab, pred, 'input', 'label', 'prediction')

                    # for metrics (remove unnecessary first dimension)
                    all_labels = torch.cat((all_labels, test_label.flatten().detach()))
                    all_predictions = torch.cat((all_predictions, (prediction > params.PRED_THRESHOLD).int().flatten().detach()))
           
            prog_bar.close()
            # average loss per city
            message = f'Valdiation {city}, avg loss: {round(avg_loss/ len(dataloader),2)}\n'
            print(message)

        # calculate, print and write metrics, return f1 score
        return self.calculate_metrics(all_labels, all_predictions, self.val_output, mode='Validation')

    def test(self):
        '''
        Start testing.
        '''
        # set model to validation mode
        self.model.eval()
        dataloader = self.test_loader

        # statistics 
        avg_loss = 0.0
        all_labels = torch.tensor([])
        all_predictions = torch.tensor([])
        #inp = None
        #pred = None
        #lab = None

        with torch.no_grad():
            for data in dataloader: # go through data in each data loader

                test_input = data[self.band]
                test_label = data['label']
                #print(f'label: {test_label.shape}, type {type(test_label)}')

                prediction = self.model.forward(test_input) # forward pass
                loss = self.criterion(prediction, test_label) # calculate error
                avg_loss += loss.item()

                # for visualization
                #inp = test_input[0]
                #pred = prediction[0]
                #lab = test_label[0]

                # for metrics (remove unnecessary first dimension)
                all_labels = torch.cat((all_labels, test_label.flatten().detach()))
                all_predictions = torch.cat((all_predictions, (prediction > params.PRED_THRESHOLD).int().flatten().detach()))

        # average loss per city
        message = f'Test Berlin, avg loss: {round(avg_loss/ len(dataloader),2)}'
        print(message)

        # DELETE
        #visualize_test(inp, lab, pred, 'input', 'label', 'prediction')

        # calculate, print and write metrics, return f1 score
        return self.calculate_metrics(all_labels, all_predictions, self.val_output, mode='Test')

    def calculate_metrics(self, labels, predictions, output, mode):
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
        # convert to ints
        labels = labels.to(torch.int)
        predictions = predictions.to(torch.int)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)

        # print and save metrics
        message = self.description + f'\n{mode} accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1 score: {f1:.2f}\n'
        print(message)
        write_results(message + '\n', output)

        # measure, hyperparameter optimization is performed on
        return f1


