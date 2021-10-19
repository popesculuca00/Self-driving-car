from imgaug.augmenters.arithmetic import AdditiveGaussianNoise
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from imgaug import augmenters as iaa
from torchvision.transforms import ColorJitter
import h5py
import numpy as np
from torchvision.io import read_image
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

class CarlaDataset(Dataset):
    def __init__(self, target_dir, for_training=True):
        super(CarlaDataset, self).__init__()
        csv = pd.read_csv(target_dir)[:600]
        self.speeds = csv["Speed"].values
        self.targets = csv[["Steer", "Gas", "Brake"]].values
        self.imgs = csv["Target path"].values
        self.commands = csv["High level command"].values
        self.trsfs = get_transforms(for_training)

    def __getitem__(self, index):
        img_orig = np.asarray( Image.open(self.imgs[index]), dtype=np.float32)
        img = self.trsfs(img_orig)
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

def get_transforms(for_training=True):
    
    if for_training:
        trsfs = transforms.Compose([
                iaa.Sequential([
                        iaa.Sometimes(0.15, [iaa.GaussianBlur((0.0, 0.75))] ),
                        iaa.Sometimes(0.2, [iaa.AdditiveGaussianNoise(scale=(0.0, 0.02))]),
                        iaa.Sometimes( 0.4,[
                            iaa.OneOf([
                                iaa.Dropout( (0.0, 0.05), per_channel=0.01) ,
                                iaa.CoarseDropout((0.0,0.1), size_percent=(0.08, 0.2), per_channel=0.5) ])] ) 
                        ]).augment_image,
                transforms.ToTensor()

        ])
           
                #
    else:
        trsfs = transforms.Compose([
                    transforms.ToTensor()
        ])
    return trsfs

if __name__ == "__main__":
    ds = CarlaDataset("data\data.csv", for_training=True)
    dl = DataLoader(dataset =ds, num_workers=0, batch_size=2)
    for index, (img, speed, target_vec, mask_vec) in enumerate(dl):
        pass
    print(img.shape)
    img = img[1].permute(  1, 2, 0)
    plt.imshow(img.numpy()/255)