import torch.nn as nn
import torch.optim as optim

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
        self.criterion = nn.CrossEntropyLoss()
    
    def training(self):
        for epoch in range(self.epochs): # all epochs

            for city_loader in self.train_loader.keys: # go through the dictionary train loader

                for i, data in enumerate(city_loader): # go through data in each data loader

                    input = data[i][self.band]
                    labels = self.model.forward(data[i]['label'])

                    predictions = self.model.forward(input) # forward pass
                    loss = self.criterion(predictions, labels) # calculate error

                    # backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    



