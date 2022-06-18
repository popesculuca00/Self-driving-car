import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=False):
        super(Conv, self).__init__()
        layers = []
        if pad:
            layers.append(nn.ZeroPad2d(pad)) 
        layers+= [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        ]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=False, max_pool=True, dropout=0.2):
        super(ConvBlock, self).__init__()
        layers = [
            Conv(in_channels, in_channels, kernel_size, stride, pad=pad),
            Conv(in_channels, out_channels, kernel_size, stride, pad=pad)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if max_pool:
            layers.append(nn.MaxPool2d((2,2)))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, do =0.2):
        super(FCBlock, self).__init__()
        layers = [
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),
            nn.LeakyReLU()
        ]
        if do:
            layers.append(nn.Dropout(do))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class FinalBlock(nn.Module):
    def __init__(self, in_features):
        super(FinalBlock, self).__init__()
        self.layers= nn.Sequential(
            nn.Linear(in_features, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class JunctionDetector(nn.Module):
    def __init__(self):
        super(JunctionDetector, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, (5,5), 1, (2,2,2,2) ),
            ConvBlock(32, 64, (3,3), 1, (1,1,1,1)),
            ConvBlock(64, 128, (3,3), 1, (1,1,1,1)),
            ConvBlock(128, 256, (3,3), 1, (1,1,1,1)),
            ConvBlock(256, 256, (3,3), 1, (1,1,1,1)),
            nn.Flatten(start_dim=1),
            FCBlock(3072, 256),
            FCBlock(256, 128),
            FCBlock(128, 64, False),
            FinalBlock(64)
        )

    @autocast()
    def forward(self, x):
        return self.layers(x)
