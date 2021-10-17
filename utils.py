import torch
from imgaug import augmenters as iaa

def get_batch_mask(commands):  # shape ( commands,  batch_size, params ) == (4 , batch_size, 3) 
    mask = torch.zeros(4, len(commands), 3)
    for i in range( len(commands) ):
        mask[ commands[i], i, :] = torch.ones(3)
    return mask

class AdditiveGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)