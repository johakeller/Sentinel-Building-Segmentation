'''Module for the models ConvNet and U-Net.'''

import torch
from torch import nn

class ConvNet(nn.Module):
    '''ConvNet model.'''

    def __init__(self, bands, dropout_rate, batch_norm=True):
        super(ConvNet, self).__init__()
        # select correct input dimension corresponding to number of bands
        if bands == 'all':
            self.channels_in = 4
        elif bands in ('RGB', 'NIRGB'):
            self.channels_in = 3
        else:
            self.channels_in = 1

        # with batch norm
        if batch_norm:
            self.model = nn.Sequential(
                nn.Conv2d(self.channels_in, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout_rate), # regularization
                nn.BatchNorm2d(32), # regularization
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout_rate), # regularization
                nn.BatchNorm2d(128), # regularization
                nn.Conv2d(128, 1, kernel_size=1, padding=0)
            )
        # without batch norm
        else:
            self.model = nn.Sequential(
                nn.Conv2d(self.channels_in, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout_rate), # regularization
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Dropout(dropout_rate), # regularization
                nn.Conv2d(128, 1, kernel_size=1, padding=0)
            )   

        self.name = 'ConvNet' # the model has a name

    def forward(self, x):
        return self.model(x)

class UNet(nn.Module):
    '''U-Net model.'''
    class Encoder(nn.Module):
        def __init__(self, in_dimension, out_dimension, dropout_rate):
            super(UNet.Encoder, self).__init__()
            self.conv_1 = nn.Conv2d(in_dimension, out_dimension, kernel_size=3, padding=1)
            self.conv_2 = nn.Conv2d(out_dimension, out_dimension, kernel_size=3, padding=1)
            self.batch_norm = nn.BatchNorm2d(out_dimension) # regularization
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(dropout_rate) # regularization
            self.relu = nn.ReLU(inplace=True)
            self.residual = None
        
        def forward(self, x):
            x = self.batch_norm(self.relu(self.conv_1(x)))
            self.residual = self.relu(self.conv_2(x)) # take residual before max_pool
            x = self.max_pool(self.residual)
            return x

    class Decoder(nn.Module):
        def __init__(self, in_dimension, out_dimension, dropout_rate):
            super(UNet.Decoder, self).__init__()
            self.up_conv = nn.ConvTranspose2d(in_dimension, out_dimension, kernel_size=2, stride=2) # undo the max pooling to match the corresponding dimension on the other side of the U
            self.conv_1 = nn.Conv2d(2*out_dimension, out_dimension, kernel_size=3, padding=1)
            self.conv_2 = nn.Conv2d(out_dimension, out_dimension, kernel_size=3, padding=1)
            self.dropout = nn.Dropout(dropout_rate) # regularization
            self.batch_norm = nn.BatchNorm2d(out_dimension) # regularization
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x, residual):
            x = self.batch_norm(self.up_conv(x))
            x = self.relu(self.conv_1(torch.cat([x, residual], dim=1))) 
            x = self.relu(self.conv_2(x))
            return x

    def __init__(self, bands, channels_out, dropout_rate):
        super(UNet,self).__init__()

        # select correct input dimension corresponding to number of bands
        if bands == 'all':
            channels_in = 4
        elif bands== 'RGB' or bands== 'NIRGB':
            channels_in = 3
        else:
            channels_in = 1
        self.name = 'UNet' # the model has a name
        
        # define encoder side
        self.en_block_1 = self.Encoder(channels_in, 64, dropout_rate)
        self.en_block_2 = self.Encoder(64, 128, dropout_rate)
        self.en_block_3 = self.Encoder(128, 256, dropout_rate)
        self.en_block_4 = self.Encoder(256, 512, dropout_rate)

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # define decoder side
        self.dec_block_1 = self.Decoder(1024, 512, dropout_rate)
        self.dec_block_2 = self.Decoder(512, 256, dropout_rate)
        self.dec_block_3 = self.Decoder(256, 128, dropout_rate)
        self.dec_block_4 = self.Decoder(128, 64, dropout_rate)

        # output block
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, channels_out, kernel_size=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoder
        x_1= self.en_block_1(x)
        x_2= self.en_block_2(x_1)
        x_3= self.en_block_3(x_2)
        x_4= self.en_block_4(x_3)
        
        # bottleneck
        x_b = self.bottleneck(x_4)
        
        # decoder (with residuals)
        y_1 = self.dec_block_1(x_b, self.en_block_4.residual)
        y_2 = self.dec_block_2(y_1, self.en_block_3.residual)
        y_3 = self.dec_block_3(y_2, self.en_block_2.residual)
        y_4 = self.dec_block_4(y_3, self.en_block_1.residual)

        # output
        return self.output_layer(y_4)



