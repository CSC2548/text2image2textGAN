import torch
import torchvision.transforms as transforms
import torch.utils.data as data

class Text2ImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir    
    self.transform = transform
    self.dataset = 

def __len__(self):

    return length

def __getitem__(self, idx):
