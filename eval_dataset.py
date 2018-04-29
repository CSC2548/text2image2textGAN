import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

class EvalDataset(data.Dataset):
    def __init__(self, data_dir, transform=None): 
        self.transform = transform
        self.root = data_dir
        self.dataset = []

        # load images into memory
        for subdir, dirs, files in os.walk(self.root):
            for img_path in tqdm(files):
                image = Image.open(os.path.join(self.root, img_path))
                if self.transform is not None:
                    image = self.transform(image)
                self.dataset.append(image)

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        image = self.dataset[idx]
        image = np.array(image, dtype=float)
        image = torch.FloatTensor(image)
        return image

