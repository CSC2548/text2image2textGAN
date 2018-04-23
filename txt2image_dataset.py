import os
import io
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pdb
from PIL import Image
import torch
from torch.autograd import Variable
import pdb
import torch.nn.functional as F
import pandas as pd
import torchvision.transforms as transforms

class Text2ImageDataset(Dataset):

    def __init__(self, datasetFile, transform=None, split=0):
        self.datasetFile = datasetFile
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.bboxes_df = pd.read_table('data/bounding_boxes.txt', sep=' ', header=None)
        self.image_paths_df = pd.read_table('data/images.txt', sep='\s+|\/+', header=None)   
        self.birds_caption_path_root = './data/birds_captions'
        self.h5py2int = lambda x: int(np.array(x))

    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]
        length = len(f[self.split])
        f.close()

        return length

    def crop_image(self, img, bbox):
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        return img


    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

        example_name = self.dataset_keys[idx]
        # example_name = self.dataset_keys[0]
        example = self.dataset[self.split][example_name]
        index_found = self.image_paths_df.index[self.image_paths_df[2]==(example_name[:-2]+'.jpg')].values[0]
        if index_found == None:
            print('ERROR: cannot find image index')

        # find right image bbox
        df_bbox = self.bboxes_df.iloc[[index_found]]
        bbox_x = df_bbox[1].values[0]
        bbox_y = df_bbox[2].values[0]
        bbox_w = df_bbox[3].values[0]
        bbox_h = df_bbox[4].values[0]
        
        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        
        wrong_name, wrong_example = self.find_wrong_image(example['class'])        
        wrong_image = bytes(np.array(wrong_example))
        wrong_index_found = self.image_paths_df.index[self.image_paths_df[2]==(wrong_name[:-2]+'.jpg')].values[0]
        if wrong_index_found == None:
            print('ERROR: cannot find image index')

        wrong_caption_path = wrong_name[:-3] + 'txt'
        wrong_caption = open(os.path.join(self.birds_caption_root, wrong_caption_path), 'r').readlines()[0]

        # find wrong image bbox
        index_found_wrong = self.image_paths_df.index[self.image_paths_df[2]==(wrong_name[:-2]+'.jpg')].values[0]
        if index_found_wrong == None:
            print('ERROR: cannot find wrong image')

        df_bbox_wrong = self.bboxes_df.iloc[[index_found_wrong]]
        wrong_bbox_x = df_bbox_wrong[1].values[0]
        wrong_bbox_y = df_bbox_wrong[2].values[0]
        wrong_bbox_w = df_bbox_wrong[3].values[0]
        wrong_bbox_h = df_bbox_wrong[4].values[0]

        byte_right_image = io.BytesIO(right_image)
        byte_wrong_image = io.BytesIO(wrong_image)

        right_image = Image.open(byte_right_image)
        wrong_image = Image.open(byte_wrong_image)
        
        right_image = self.crop_image(right_image, bbox=[bbox_x, bbox_y, bbox_w, bbox_h]) 
        wrong_image = self.crop_image(wrong_image, bbox=[wrong_bbox_x, wrong_bbox_y, wrong_bbox_w, wrong_bbox_h]) 

        # right_image = Image.open(byte_right_image).resize((64, 64))
        # wrong_image = Image.open(byte_wrong_image).resize((64, 64))
        #right_image128 = Image.open(byte_right_image).resize((128, 128))
        #wrong_image128 = Image.open(byte_wrong_image).resize((128, 128))

        right_image = transforms.Resize((64,64))(right_image)
        wrong_image = transforms.Resize((64,64))(wrong_image)
        
        right_image128 = transforms.Resize((128,128))(right_image)
        wrong_image128 = transforms.Resize((128,128))(wrong_image) 

        # print(right_image.size, wrong_image.size, right_image128.size, wrong_image128.size)

        right_image = self.validate_image(right_image)
        wrong_image = self.validate_image(wrong_image)

        right_image128 = self.validate_image128(right_image128)
        wrong_image128 = self.validate_image128(wrong_image128)

        txt = np.array(example['txt']).astype(str)

        sample = {
                'right_images': torch.FloatTensor(right_image),
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_images': torch.FloatTensor(wrong_image),
                'txt': str(txt),
                'right_images128': torch.FloatTensor(right_image128),
                'wrong_images128': torch.FloatTensor(wrong_image128),
                'wrong_txt': str()
                }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] =sample['wrong_images'].sub_(127.5).div_(127.5)

        sample['right_images128'] = sample['right_images128'].sub_(127.5).div_(127.5)
        sample['wrong_images128'] =sample['wrong_images128'].sub_(127.5).div_(127.5)

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example_name, example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']


    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

    def validate_image128(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((128, 128, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: dictionary with keys (right_images, ...). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x['txt']), reverse=True)
    images, captions, wrong_captions = zip(*data)

    wrong_captions = list(wrong_captions)
    wrong_captions.sort(key=lambda x: len(x), reverse=True)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # Getting wrong targets(captions)
    wrong_lengths = [len(cap) for cap in wrong_captions]
    wrong_targets = torch.zeros(len(wrong_captions), max(wrong_lengths)).long()
    for i, cap in enumerate(wrong_captions):
        end = wrong_lengths[i]
        wrong_targets[i, :end] = cap[:end]
    
    return images, targets, lengths, wrong_targets, wrong_lengths
