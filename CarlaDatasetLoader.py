from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from PIL import Image
import torch
from albumentations.augmentations.transforms import Blur, GaussianBlur, RandomBrightnessContrast # blur augs 
from albumentations.augmentations.transforms import GaussNoise # noise aug
from albumentations.augmentations.transforms import CoarseDropout, GridDropout # dropouts
from albumentations.augmentations.transforms import RandomRain, RandomFog, RandomSunFlare, RandomShadow, RandomSnow # natural effects augmentations
from albumentations import Compose, OneOf, Sequential # aug helper functions
from albumentations.pytorch import ToTensorV2 


class CarlaBaseDataset(Dataset):
    def __init__(self, target_dir):
        super(CarlaBaseDataset, self).__init__()
        csv = pd.read_csv(target_dir)[:600]
        self.speeds = csv["Speed"].values
        self.targets = csv[["Steer", "Gas", "Brake"]].values
        self.imgs = csv["Target path"].values
        self.commands = csv["High level command"].values

    def __getitem__(self, index):
        img = np.array( Image.open(self.imgs[index]), dtype=np.uint8, copy=True) #convert to uint8 for gaussian noise, copy= True for warning suppression
        target = self.targets[index]
        command = int(self.commands[index]-2)
        target_vec = np.zeros((4,3), dtype=np.float32 )
        target_vec[command, :] =  target
        speed = np.array( self.speeds[index]/90, dtype=np.float32  )  # reshape to (-1,1)
        mask_vec = np.zeros((4,3), dtype= np.float32)
        mask_vec[command, :] = np.ones((1,3), dtype=np.float32 )
        return img, speed, target_vec, mask_vec
    
    def __len__(self):
        return len(self.imgs)


class CarlaDataset(Dataset):
    def __init__(self, dataset, train_flag=True):
        super(CarlaDataset, self).__init__()
        self.dataset = dataset
        self.img_transforms = _get_transforms(train_flag)
        
    def __getitem__(self, index):
        (img, speed, target_vec, mask_vec) = self.dataset[index]

        #print( img.shape )
        img = self.img_transforms(image=img)["image"]

        speed = np.reshape(speed, (1))
        img = img / 255

        return img, np.reshape(speed, (1)), target_vec, mask_vec

    def __len__(self):
        return len(self.dataset)


def _get_transforms(train_flag=True):
    """
    Get Compose object for image augmentation
    """
    if train_flag:

        weather_transforms = Sequential([
            OneOf([
            RandomRain(rain_type="drizzle", blur_value=4, brightness_coefficient=0.99, p=1),
            RandomRain(rain_type="heavy", blur_value=4, brightness_coefficient=0.99, p=1),
            RandomRain(rain_type="torrential", blur_value=4, brightness_coefficient=0.99, p=1),
            RandomFog(fog_coef_lower=0.2, p=1),
            RandomShadow(num_shadows_lower=1, num_shadows_upper=4, p=1),
            RandomSnow()
            ], p=1)
        ], p=1)

        synthetic_transforms = Sequential([
            OneOf([
                GridDropout( ratio=0.3, unit_size_min=2, unit_size_max=7, p=0.05 ) ,
                CoarseDropout( max_holes=10, max_height=10, max_width=20, p=0.95 )
            ], p=1)
        ], p=1)

        blur_transforms = Sequential([
            OneOf([
                GaussianBlur(blur_limit=(1, 3), p=1),
                Blur(blur_limit=4, p=1)
            ], p=1),
        ], p=1)

        trsfs = Compose([

                RandomBrightnessContrast(brightness_limit=(-0.15, 0.25), contrast_limit=0.1,  p=0.4),
                GaussNoise(var_limit= (10,50), p=0.2), 
                OneOf([
                    weather_transforms,
                    synthetic_transforms,
                    blur_transforms
                ], p=0.5),
                ToTensorV2()
        ])
    else:
        trsfs = Compose([
                ToTensorV2()
        ])
    return trsfs


def _base_dataset(target_dir="data\data.csv", for_training=True):
    """
    Returns carla base dataset
    """
    return CarlaBaseDataset(target_dir)

def dataset_to_dataloader(dataset=None, num_workers=1, batch_size=32, shuffle=True, pin_memory="auto"):
    """
    Returns batch of img, speed, target_vec, mask_vec
    """
    if pin_memory == "auto":
        pin_memory = True if torch.cuda.is_available() else False

    if dataset is None:
        dataset = _base_dataset()
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory)


def get_dataloaders(base_path="data/data.csv", train_size = 0.95, num_workers=1, batch_size=32, shuffle=True, pin_memory="auto"):

    assert (train_size <= 1.0 and train_size>0.0), "Training to validation ratio must be a float between 0 and 1"
    
    base_dataset = CarlaBaseDataset(base_path)

    if train_size < 1.0 :
        
        data_len = len(base_dataset)
        valid_indexes = np.random.choice(data_len, int((1-train_size)*data_len) )
        train_indexes = np.setdiff1d( np.arange(data_len), valid_indexes, assume_unique=True  )

        train_subset = torch.utils.data.Subset(base_dataset, train_indexes)
        valid_subset = torch.utils.data.Subset(base_dataset, valid_indexes)

        train_ds = CarlaDataset(train_subset, True)
        valid_ds = CarlaDataset(valid_subset, False)

        train_dl = dataset_to_dataloader(train_ds, num_workers, batch_size, shuffle, pin_memory)
        valid_dl = dataset_to_dataloader(valid_ds, num_workers, batch_size, False, pin_memory)
        return train_dl, valid_dl

    if train_size == 1:
        return dataset_to_dataloader(base_dataset, num_workers, batch_size, shuffle, pin_memory)
