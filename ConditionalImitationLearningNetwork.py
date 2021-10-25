import os
import torch
import torch.nn as nn

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
        self.do = nn.Dropout(0.3)
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
        self.do_1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, 128)
        self.do_2 = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu ( self.do_1( self.fc_1(x) ) )
        x = self.do_2( self.fc_2(x) )
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


class BranchBlock(nn.Module):
    def __init__(self, target_direction):
        super(BranchBlock, self).__init__()
        self.target_direction = target_direction
        self.fc_1 = nn.Linear(512, 256)
        self.do = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 3)           ### steer, gas, brake


    def forward(self, x):
        x = self.do( self.fc_1(x) )
        x = self.relu( x )
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x

    @property
    def direction(self):
        return self.target_direction
        
class ImitationLearningNetwork_Training(nn.Module):
    def __init__(self):
        super(ImitationLearningNetwork_Training, self).__init__()
        self.vision_module = VisionModule()  # out : 512
        self.speed_module  = SpeedModule()   # out : 128
        self.concat_module = ConcatenationModule() # out : 512
        self.branch_commands = ["NoInput", "left", "right", "forward"]
        self.branches = nn.ModuleList([BranchBlock(i) for i in self.branch_commands])
    
    def forward(self, image_input, speed_input):
        img_embedding = self.vision_module(image_input)
        speed_embedding = self.speed_module(speed_input)
        x = self.concat_module(img_embedding, speed_embedding)
        preds = torch.stack( [ branch(x) for branch in self.branches ], dim=1 )

        return preds # shape ( commands,  batch_size, params ) == (4 , batch_size, 3) 


def get_learning_rate_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode = "min",
        factor = 0.5,
        patience=10,
        cooldown=5,
        verbose=False
    )
    return scheduler