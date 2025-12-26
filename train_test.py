'''Module to run training, validation, and test.'''

import os
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import params

class IoU(nn.Module):
    '''
    Class to compute IoU loss.
    '''

    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, predictions, labels):
        
        # apply Sigmoid
        predictions = torch.sigmoid(predictions).to(params.DEVICE)

        # flattening
        predictions = torch.flatten(predictions)
        labels= torch.flatten(labels)

        # intersection where both are 1
        intersection = (predictions* labels)
        intersection = intersection.sum()

        # union: remove the intersection from sum
        union = (predictions + labels).sum()- intersection

        # avoid zero division
        return 1.0 -((intersection + 1.0)/ (union + 1.0))

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

    def __init__(self, model, train_loader=None, val_loader=None, test_loader=None, mode = 'train', epochs=params.EPOCHS, lr = None, train_output=None, val_output=None, band=None, weight_decay=None, model_name=None, lr_scheduler=False, iou_w=0.0, bce_w=1.0):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mode = mode
        self.epochs = epochs
        self.band = band
        self.description = f'{model_name}, bands: {band}, learning rate: {lr}, weight decay: {weight_decay}'
        self.model = model.to(params.DEVICE) # initialize model after bands, because number of channels has to be adjusted
        self.train_output = train_output
        self.val_output = val_output
        self.learning_rate = lr
        self.optimizer = optim.Adam(model.parameters(), lr =self.learning_rate, weight_decay=weight_decay)
        if lr_scheduler:
            self.lr_scheduler = StepLR(self.optimizer, step_size = 8, gamma = 0.95) 
        else:
            self.lr_scheduler = None
        self.iou = IoU() 
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.iou_f = iou_w
        self.bce_f = bce_w
    
    def training(self):
        '''
        Start training.
        '''

        # print hyperparameters
        print(self.description)

        # initialize metrics
        f1 = 0
        accuracy = 0
        precision = 0
        recall = 0
        sched_ctr = 0

        # set model to train mode
        self.model.train()
        for epoch in range(self.epochs): # all epochs
            
            # collect labels and predictions
            all_labels = torch.Tensor([]).to(params.DEVICE)
            all_predictions = torch.Tensor([]).to(params.DEVICE)
            # increase counter
            sched_ctr += 1

            for city, dataloader in self.train_loader.items(): # go through the dictionary train loader (for each city)

                # statistics 
                prog_bar = tqdm(total=len(dataloader.dataset), desc=f'Training epoch {epoch+1}, {city}', position=0, leave=False)
                avg_loss = 0.0

                for data in dataloader: # go through data in each data loader

                    # update prog_bar
                    prog_bar.update(dataloader.batch_size)

                    train_input = data[self.band].to(params.DEVICE)
                    train_label = data['label'].to(params.DEVICE)
                    
                    prediction = self.model.forward(train_input) # forward pass

                    # for UNet:compund loss BCE and IoU
                    loss = self.bce_f*self.criterion(prediction, train_label) + self.iou_f*self.iou(prediction, train_label) # calculate error
                    avg_loss += loss.item()

                    # backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # for metrics (remove first dimension)
                    all_labels = torch.cat((all_labels, train_label.flatten().detach()))
                    # create binary predictions out of probabilities by thresholding
                    all_predictions = torch.cat((all_predictions, (prediction > params.PRED_THRESHOLD).flatten().detach()))

                    # lr scheduler step, if lr scheduler exists
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                prog_bar.close()             
                # average loss after each city
                message = f'Training epoch {epoch + 1}, {city}, avg loss: {round(avg_loss/ len(dataloader),2)}\n'
                print(message)
 
            # calculate, metrics, return f1 score
            f1_tmp, accuracy_tmp, precision_tmp, recall_tmp = self.calculate_metrics(all_labels, all_predictions)

            f1 += f1_tmp
            accuracy += accuracy_tmp
            precision += precision_tmp
            recall += recall_tmp
        
        # print and save metrics
        f1 /= self.epochs
        accuracy /= self.epochs
        precision /= self.epochs
        recall /= self.epochs
        
        message = self.description + f'\nTraining accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1 score: {f1:.2f}\n'
        print(message)
        write_results(message + '\n', self.train_output)

        return f1

    def validation(self):
        '''
        Start validation.
        '''

        # set model to validation mode
        self.model.eval()

        # initialize metrics
        all_labels = torch.tensor([]).to(params.DEVICE)
        all_predictions = torch.tensor([]).to(params.DEVICE)

        for city, dataloader in self.val_loader.items(): # go through the dictionary validation loader (for each city)

            # statistics 
            avg_loss = 0.0
            prog_bar = tqdm(total=len(dataloader.dataset), desc=f'{city} validation', position=0, leave=False)

            with torch.no_grad():
                for i, data in enumerate(dataloader): # go through data in each data loader
                    # update progress prog_bar
                    prog_bar.update(dataloader.batch_size)

                    test_input = data[self.band].to(params.DEVICE)
                    test_label = data['label'].to(params.DEVICE)

                    prediction = self.model.forward(test_input) # forward pass
                    loss = self.bce_f*self.criterion(prediction, test_label) + self.iou_f*self.iou(prediction, test_label)
                    avg_loss += loss.item()

                    # for metrics (remove unnecessary first dimension)
                    all_labels = torch.cat((all_labels, test_label.flatten().detach()))
                    all_predictions = torch.cat((all_predictions, (prediction > params.PRED_THRESHOLD).int().flatten().detach()))
           
            prog_bar.close()
            
        # calculate, metrics, return f1 score
        f1, accuracy, precision, recall = self.calculate_metrics(all_labels, all_predictions)
    
        message = self.description + f'\nValidation accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1 score: {f1:.2f}\n'
        print(message)
        write_results(message + '\n', self.val_output)

        return f1

    def test(self):
        '''
        Start testing.
        '''

        # set model into validation mode
        self.model.eval()
        dataloader = self.test_loader

        # statistics 
        avg_loss = 0.0
        all_labels = torch.tensor([]).to(params.DEVICE)
        all_predictions = torch.tensor([]).to(params.DEVICE)

        with torch.no_grad():
            for data in dataloader: # go through data in each data loader

                test_input = data[self.band].to(params.DEVICE)
                test_label = data['label'].to(params.DEVICE)

                prediction = self.model.forward(test_input) # forward pass
                loss = loss = self.bce_f*self.criterion(prediction, test_label) + self.iou_f*self.iou(prediction, test_label)
                avg_loss += loss.item()

                # for metrics (remove unnecessary first dimension)
                all_labels = torch.cat((all_labels, test_label.flatten().detach()))
                all_predictions = torch.cat((all_predictions, (prediction > params.PRED_THRESHOLD).int().flatten().detach()))

        # average loss per city
        message = f'Test Berlin, avg loss: {round(avg_loss/ len(dataloader),2)}'
        print(message)

        # calculate, metrics, return f1 score
        f1, accuracy, precision, recall = self.calculate_metrics(all_labels, all_predictions)
    
        message = self.description + f'\nTest accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1 score: {f1:.2f}\n'
        print(message)
        write_results(message + '\n', self.val_output)

        return f1

    def calculate_metrics(self, labels, predictions):
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
        labels = labels.cpu().to(torch.int).numpy()
        predictions = predictions.cpu().to(torch.int).numpy()

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)

        # measure, hyperparameter optimization is performed on
        return f1, accuracy, precision, recall


