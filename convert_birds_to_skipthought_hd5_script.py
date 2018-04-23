import os
from os.path import join, isfile
import numpy as np
import h5py
from glob import glob
from torch.utils.serialization import load_lua  
from PIL import Image 
import yaml
import io
import pdb
import nltk
import re
from collections import Counter
import torch
from torch.autograd import Variable
import sys
sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip
dir_st = 'data/skip-thoughts'
import tqdm
import pickle



with open('config.yaml', 'r') as f:
    config = yaml.load(f)

images_path = config['birds_images_path']
embedding_path = config['birds_embedding_path']
text_path = config['birds_text_path']
datasetDir = config['birds_dataset_path']

val_classes = open(config['val_split_path']).read().splitlines()
train_classes = open(config['train_split_path']).read().splitlines()
test_classes = open(config['test_split_path']).read().splitlines()

f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def sanitize_string(caption):
    caption = caption.strip()
    caption = caption.encode('ascii', 'ignore')
    caption = caption.decode('ascii')
    caption = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', '', caption)
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    return tokens

def build_vocab():

    threshold = 4
    """Build a simple vocabulary wrapper."""
    # go through all files
    counter = Counter()
    for _class in sorted(os.listdir(embedding_path)):
        txt_path = os.path.join(text_path, _class)
        for txt_file in sorted(glob(txt_path + "/*.txt")):
            lines = open(txt_file, "r").readlines()
            for caption in lines:
                tokens = sanitize_string(caption)
                counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

# vocab = build_vocab()
# with open('data/birds_vocab.pkl', 'wb') as f:
#     pickle.dump(vocab, f)

with open('data/birds_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


all_words_in_vocab = vocab.word2idx.keys()
uniskip = UniSkip(dir_st, all_words_in_vocab)


def get_ids(tokens, vocab):
    ids = []
    # appending start and eos at the beginning and the end respectively for every sequence
    ids.append(vocab('<start>'))
    for word in tokens:
        ids.append(vocab(word))
    ids.append(vocab('<end>'))
    return ids

for _class in sorted(os.listdir(embedding_path)):
    split = ''
    if _class in train_classes:
        split = train
    elif _class in val_classes:
        split = valid
    elif _class in test_classes:
        split = test

    data_path = os.path.join(embedding_path, _class)
    txt_path = os.path.join(text_path, _class)
    for example, txt_file in zip(sorted(glob(data_path + "/*.t7")), sorted(glob(txt_path + "/*.txt"))):
        example_data = load_lua(example)
        img_path = example_data['img']
        embeddings = example_data['txt'].numpy()
        example_name = img_path.split('/')[-1][:-4]

        f = open(txt_file, "r")
        txt = f.readlines()
        f.close()

        img_path = os.path.join(images_path, img_path)
        img = open(img_path, 'rb').read()

        txt_choice = np.random.choice(range(10), 5)

        embeddings = embeddings[txt_choice]
        
    
        txt = np.array(txt)
        txt = txt[txt_choice]
        dt = h5py.special_dtype(vlen=str)


        batch_txt_ids = []
        max_len = 0
        id_lens = []
        for t in txt:
            skip_thought_txt  = sanitize_string(t)
            txt_ids = get_ids(skip_thought_txt, vocab)
            max_len = len(txt_ids) if max_len < len(txt_ids) else max_len
            batch_txt_ids.append(txt_ids)
            id_lens.append(len(txt_ids))

        # padding with eos
        for arr in batch_txt_ids:
            n = len(arr)
            rem = max_len - n
            concat_arr = [vocab('<end>')]*rem
            arr+=concat_arr


        input = Variable(torch.LongTensor(batch_txt_ids))
        output_seq2vec = uniskip(input, lengths=id_lens).data.numpy()
    
        # for c, e in enumerate(embeddings):
        for c, e in enumerate(output_seq2vec):
    
            ex = split.create_group(example_name + '_' + str(c))
            ex.create_dataset('name', data=example_name)
            ex.create_dataset('img', data=np.void(img))
            ex.create_dataset('embeddings', data=e)
            ex.create_dataset('class', data=_class)
            ex.create_dataset('txt', data=txt[c].astype(object), dtype=dt)

        print(example_name)


