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
from build_vocab import *
import nltk

class Text2ImageDataset(Dataset):

    def __init__(self, datasetFile, dataset_type, vocab, transform=None, split=0):
        self.datasetFile = datasetFile
        self.dataset_type = dataset_type
        self.vocab = vocab
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.bboxes_df = pd.read_table('data/bounding_boxes.txt', sep=' ', header=None)
        self.image_paths_df = pd.read_table('data/images.txt', sep='\s+|\/+', header=None)
        self.birds_caption_path_root = './data/birds_captions/'
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

        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)

        wrong_name, wrong_example, wrong_txt = self.find_wrong_image(example['class']) 
        wrong_image = bytes(np.array(wrong_example))
        if self.dataset_type=='birds':
            wrong_index_found = self.image_paths_df.index[self.image_paths_df[2]==(example_name[:-2]+'.jpg')].values[0]
            if wrong_index_found == None:
                print('ERROR: cannot find image index')
            wrong_txt_sub_path = self.image_paths_df.iloc[[wrong_index_found]][1].values[0] + '/' + self.image_paths_df.iloc[[wrong_index_found]][2].values[0][:-3] + 'txt'
            wrong_txt_path = self.birds_caption_path_root + wrong_txt_sub_path
            wrong_txt = open(wrong_txt_path, 'r').readlines()[0]

            index_found = self.image_paths_df.index[self.image_paths_df[2]==(example_name[:-2]+'.jpg')].values[0]
            if index_found == None:
                print('ERROR: cannot find image index')

            # find right image bbox
            df_bbox = self.bboxes_df.iloc[[index_found]]
            bbox_x = df_bbox[1].values[0]
            bbox_y = df_bbox[2].values[0]
            bbox_w = df_bbox[3].values[0]
            bbox_h = df_bbox[4].values[0]
        
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

        if self.dataset_type == 'birds':
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

        # preprocess txt and wrong_txt
        txt = str(txt)
        txt = txt.strip()
        txt = txt.encode('ascii', 'ignore')
        txt = txt.decode('ascii')
        exclude = set(string.punctuation)
        preproc_txt = ''.join(ch for ch in txt if ch not in exclude)
        tokens = nltk.tokenize.word_tokenize(preproc_txt.lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        caption = torch.Tensor(caption)


        wrong_txt = wrong_txt.strip()
        wrong_txt = wrong_txt.encode('ascii', 'ignore')
        wrong_txt = wrong_txt.decode('ascii')
        exclude = set(string.punctuation)
        preproc_wrong_txt = ''.join(ch for ch in wrong_txt if ch not in exclude)
        wrong_tokens = nltk.tokenize.word_tokenize(preproc_wrong_txt.lower())
        wrong_caption = []
        wrong_caption.append(self.vocab('<start>'))
        wrong_caption.extend([self.vocab(token) for token in wrong_tokens])
        wrong_caption.append(self.vocab('<end>'))
        wrong_caption = torch.Tensor(wrong_caption)

        sample = {
                'right_images': torch.FloatTensor(right_image),
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_images': torch.FloatTensor(wrong_image),
                'caption': caption,
                'txt': txt,
                'right_images128': torch.FloatTensor(right_image128),
                'wrong_images128': torch.FloatTensor(wrong_image128),
                'wrong_caption': wrong_caption
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
            return example_name, example['img'], str(np.array(example['txt']))

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
        data: dictionary with keys (right_images, right_embed, wrong_images, caption, 
                                    right_images128, wrong_images128, wrong_caption). 

    Returns:
        sample: dictionary with keys sorted by len(caption) for right images/embed etc,
                                   and by len(wrong_caption) for wrong images, wrong_caption.
    """

    collate_data = {}

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x['caption'].size(0), reverse=True)

    captions = []
    right_images = []
    right_embeds = []
    wrong_images = []
    right_images128 = []
    wrong_images128 = []
    collate_data['txt'] = []
    for i in range(len(data)):
        collate_data['txt'].append(data[i]['txt'])
        captions.append(data[i]['caption'])
        right_images.append(data[i]['right_images'])
        right_embeds.append(data[i]['right_embed'])
        wrong_images.append(data[i]['wrong_images'])
        right_images128.append(data[i]['right_images128'])
        wrong_images128.append(data[i]['wrong_images128'])
         

    # sort and get captions, lengths, images, embeds etc
    lengths = [len(cap) for cap in captions]
    collate_data['lengths'] = lengths
    collate_data['captions'] = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        collate_data['captions'][i, :end] = cap[:end]

    collate_data['right_images'] = torch.stack(right_images, 0)
    collate_data['right_embed'] = torch.stack(right_embeds, 0)
    collate_data['wrong_images'] = torch.stack(wrong_images, 0)
    collate_data['right_images128'] = torch.stack(right_images128, 0)
    collate_data['wrong_images128'] = torch.stack(wrong_images128, 0)

    # sort and get wrong_captions, wrong_lengths
    wrong_captions = []
    wrong_lengths = []
    for i in range(len(data)):
         wrong_captions.append(data[i]['wrong_caption'])
    wrong_captions.sort(key=lambda x: len(x), reverse=True)
    wrong_lengths = [len(cap) for cap in wrong_captions]    
    collate_data['wrong_lengths'] = wrong_lengths
    collate_data['wrong_captions'] = torch.zeros(len(wrong_captions), max(wrong_lengths)).long()
    for i, cap in enumerate(wrong_captions):
        end = wrong_lengths[i]
        collate_data['wrong_captions'][i, :end] = cap[:end]
   
 
    return collate_data
