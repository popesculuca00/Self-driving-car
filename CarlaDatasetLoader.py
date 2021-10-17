from imgaug.augmenters.arithmetic import AdditiveGaussianNoise
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from imgaug import augmenters as iaa
from torchvision.transforms.transforms import ColorJitter
import h5py
import numpy as np
from torchvision.io import read_image
import pandas as pd
import torch

class CarlaDataset(Dataset):
    def __init__(self, target_dir, is_train=True):
        #super(CarlaDataset, self).__init__()
        csv = pd.read_csv(target_dir)[:20]
        self.speeds = csv["Speed"].values
        self.targets = csv[["Steer", "Gas", "Brake"]].values
        self.imgs = csv["Target path"].values
        self.commands = csv["High level command"].values
        self.is_train = is_train
        
        self.init_transforms()
    
    def init_transforms(self):
        if self.is_train:

            seq = iaa.Sequential([
                iaa.Sometimes( 0.5, iaa.GaussianBlur((0.0, 1.5)) ),
                iaa.Sometimes( 0.5, iaa.AdditiveGaussianNoise(scale=(0.0, 2))),
                iaa.OneOf([ iaa.Dropout( (0.0, 1.0), per_channel=0.05),
                            iaa.CoarseDropout(0.005, size_percent=0.1)])
            ])

            self.transforms = transforms.Compose([
                transforms.Lambda(lambda x: seq(x))
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
                ])


    def __getitem__(self, index):

        img = read_image(self.imgs[index])
        img = self.transforms(img)
        target = self.targets[index]
        command = int(self.commands[index]-2)
        target_vec = np.zeros((4,3), dtype=np.float32 )
        target_vec[command, :] =  target

        speed = np.array( self.speeds[index]/90  )
        mask_vec = np.zeros((4,3), dtype= np.float32)
        mask_vec[command, :] = np.ones((1,3), dtype=np.float32 )

        return img, speed, target_vec, mask_vec#.reshape(-1),

    
    def __len__(self):
        return len(self.imgs)

def get_train_augmenters():
    aug = iaa.Sequential([
        
    ])


def get_dataloader( target_dir = "data\data.csv", num_workers=8, batch_size=32, is_train=True):
    dataset = CarlaDataset(target_dir, is_train=is_train)

    if is_train:
        image_augmenters = get_train_augmenters()
    
    
    dataloader = DataLoader(dataset, augmenters=image_augmenters, num_workers=num_workers, batch_size=batch_size, shuffle=True)

                #     iaa.GaussianBlur((0.0, 1.5)),
                #     iaa.AdditiveGaussianNoise(scale=(0.0, 0.05)),
                #     iaa.Dropout( (0.0, 1.0), per_channel=0.05) ,
                #     iaa.CoarseDropout(0.005, size_percent=0.1)]),
                # transforms.ColorJitter(brightness= 0.35 ),