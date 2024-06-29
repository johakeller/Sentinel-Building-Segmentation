import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, bands):
        super(ConvNet, self).__init__()
        # select correct input dimension corresponding to number of bands
        if bands == 'all':
            self.channels_in = 4
        elif bands== 'RGB' or bands== 'NIRGB':
            self.channels_in = 3
        else:
            self.channels_in = 1

        self.model = nn.Sequential(
            nn.Conv2d(self.channels_in, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.model(x)