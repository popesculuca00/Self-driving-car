import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class ConvBlock(nn.Module):
    def __init__(self, input_filters, n_filters, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_filters, n_filters, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.bn( self.conv(x) )
        return x


class ImageBlockFC(nn.Module):
    def __init__(self):
        super(ImageBlockFC, self).__init__()
        self.fc_1 = nn.Linear(8192, 512)
        self.do = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu( self.do(x) )
        x = self.fc_2(x)
        return x


class VisionModule(nn.Module):
    def __init__(self):
        super(VisionModule, self).__init__()
        self.block_1 = ConvBlock(3, 32, 5, 2)
        self.block_2 = ConvBlock(32, 32, 3)
        self.block_3 = ConvBlock(32, 64, 3, 2)
        self.block_4 = ConvBlock(64, 64, 3)
        self.block_5 = ConvBlock(64, 128, 3, 2)
        self.block_6 = ConvBlock(128, 128, 3)
        self.block_7 = ConvBlock(128, 256, 3)
        self.block_8 = ConvBlock(256, 256, 3)
        self.relu = nn.ReLU()
        self.fc_img = ImageBlockFC()
        self.flatten_layer = nn.Flatten(start_dim=1)
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.relu ( self.block_2(x) )
        x = self.relu ( self.block_3(x) )
        x = self.relu ( self.block_4(x) )
        x = self.relu ( self.block_5(x) )
        x = self.relu ( self.block_6(x) )
        x = self.relu ( self.block_7(x) )
        x = self.relu ( self.block_8(x) )
        x = self.flatten_layer(x)
        x = self.fc_img(x)
        return x


class SpeedModule(nn.Module):
    def __init__(self):
        super(SpeedModule, self).__init__()
        self.fc_1 = nn.Linear(1, 128)
        self.do_1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu ( self.do_1( self.fc_1(x) ) )
        return x


class ConcatenationModule(nn.Module):
    def __init__(self):
        super(ConcatenationModule, self).__init__()
        self.fc = nn.Linear(512 + 128, 512)
        self.do = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, image_embedding, speed_embedding):
        x = torch.cat([ image_embedding, speed_embedding ], dim=1)
        x = self.do( self.fc(x) )
        x = self.relu(x)
        return x


class JointPreprocessingBlock(nn.Module):
    def __init__(self):
        super(JointPreprocessingBlock, self).__init__()
        do_val = 0.3
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(512, 512)
        self.do_1 = nn.Dropout(do_val)
        self.fc_2 = nn.Linear(512, 512)
        self.do_2 = nn.Dropout(do_val)
        self.fc_3 = nn.Linear(512, 512)
        self.do_3 = nn.Dropout(do_val)
        self.fc_4 = nn.Linear(512, 512)
        self.do_4 = nn.Dropout(do_val)

    def forward(self, x):
        x = self.relu(self.do_1(self.fc_1(x)))
        x = self.relu(self.do_2(self.fc_2(x)))
        x = self.relu(self.do_3(self.fc_3(x)))
        x = self.relu(self.do_4(self.fc_4(x)))
        return x


class BranchBlock(nn.Module):
    def __init__(self):
        super(BranchBlock, self).__init__()
        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_3 = nn.Linear(512, 512)
        self.fc_4 = nn.Linear(512, 512)
        self.steer = nn.Linear(512, 1)
        self.gas = nn.Linear(512, 1)
        self.brake = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.tanh = torch.tanh
    
    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        steer = self.tanh( self.steer(x) )
        gas = self.relu( self.gas(x) )
        brake = self.relu( self.brake(x) )
        return torch.cat([steer, gas, brake], dim=1)


class ConditionalBranchModel(nn.Module):
    def __init__(self):
        super(ConditionalBranchModel, self).__init__()
        self.vision_module = VisionModule()
        self.speed_module  = SpeedModule()
        self.concat_module = ConcatenationModule()
        self.joint_preproc_module = JointPreprocessingBlock()
        self.branch = BranchBlock()

    @torch.no_grad()
    @autocast()
    def forward(self, image_input, speed_input):
        img_embedding = self.vision_module(image_input)
        speed_embedding = self.speed_module(speed_input)
        x = self.concat_module(img_embedding, speed_embedding)
        x = self.joint_preproc_module(x)
        x = self.branch(x)
        return x        
  