from imgaug.augmenters.arithmetic import AdditiveGaussianNoise
from imgaug.augmenters.contrast import GammaContrast
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import imgaug
from imgaug import augmenters as iaa
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms.transforms import ToPILImage

class CarlaBaseDataset(Dataset):
    def __init__(self, target_dir):
        super(CarlaBaseDataset, self).__init__()
        csv = pd.read_csv(target_dir)[:600]
        self.speeds = csv["Speed"].values
        self.targets = csv[["Steer", "Gas", "Brake"]].values
        self.imgs = csv["Target path"].values
        self.commands = csv["High level command"].values
        #self.trsfs = get_transforms(for_training)

    def __getitem__(self, index):
        img = np.asarray( Image.open(self.imgs[index]), dtype=np.float32)
        #img = self.trsfs(img_orig)
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
        img = self.img_transforms(img)
        return img, speed, target_vec, mask_vec


    def __len__(self):
        return len(self.dataset)


def _get_transforms(train_flag=True):
    """
    Get Compose object for image augmentation
    """
    if train_flag:
        trsfs = transforms.Compose([
                iaa.Sequential([
                        iaa.Sometimes(0.7, [iaa.GammaContrast((0.8, 1.15))] ),
                        iaa.Sometimes(0.2, [iaa.GaussianBlur((0.0, 0.75))] ),
                        iaa.Sometimes(0.6, [iaa.AdditiveGaussianNoise(scale=(0.0, 0.02))]),
                        iaa.Sometimes( 0.4,[
                            iaa.OneOf([
                                iaa.Dropout( (0.0, 0.05), per_channel=0.01) ,
                                iaa.CoarseDropout((0.0,0.1), size_percent=(0.08, 0.2), per_channel=0.5) ])] ) 
                        ]).augment_image,
                transforms.ToTensor()

        ])
    else:
        trsfs = transforms.Compose([
                transforms.ToTensor()
        ])
    return trsfs

def _worker_seed_initializer(worker_id):
    """
    Randomizes augmentation seed for each individual worker
    """
    imgaug.seed(np.random.get_state()[1][0] + worker_id )

def _base_dataset(target_dir="data\data.csv", for_training=True):
    """
    Returns carla base dataset
    """
    return CarlaBaseDataset(target_dir)

def dataset_to_dataloader(dataset=None, num_workers=1, batch_size=32, shuffle=True, worker_init_fn=_worker_seed_initializer, pin_memory="auto"):
    """
    Returns batch of img, speed, target_vec, mask_vec
    """
    if pin_memory == "auto":
        pin_memory = True if torch.cuda.is_available() else False

    if dataset is None:
        dataset = _base_dataset()
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn, shuffle=True, pin_memory=pin_memory)


def get_dataloaders(base_path="data/data.csv", train_size = 0.95, num_workers=1, batch_size=32, shuffle=True, worker_seed_initializer=_worker_seed_initializer, pin_memory="auto"):

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

        train_dl = dataset_to_dataloader(train_ds, num_workers, batch_size, shuffle, worker_seed_initializer, pin_memory)
        valid_dl = dataset_to_dataloader(valid_ds, num_workers, batch_size, shuffle, worker_seed_initializer, pin_memory)
        return train_dl, valid_dl

    if train_size == 1:
        return dataset_to_dataloader(base_dataset, num_workers, batch_size, shuffle, worker_seed_initializer, pin_memory)
