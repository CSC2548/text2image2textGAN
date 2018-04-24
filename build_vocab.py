import nltk
import pickle
import argparse
from collections import Counter
import os
import pdb
from tqdm import tqdm
import re
import string

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

def build_vocab(filepath, threshold):
    """Build a simple vocabulary wrapper."""
    # go through all files
    counter = Counter()
    for subdir, dirs, files in os.walk(filepath):
        for caption_file in tqdm(files):
            if caption_file[-3:] != 'txt':
                continue
            with open(os.path.join(subdir, caption_file), 'r') as f:
                captions = f.readlines()
                for caption in captions:
                    caption = caption.strip()
                    caption = caption.encode('ascii', 'ignore')
                    caption = caption.decode('ascii')
                    exclude = set(string.punctuation)
                    preproc_caption = ''.join(ch for ch in caption if ch not in exclude)
                    tokens = nltk.tokenize.word_tokenize(preproc_caption.lower())
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

def main(args):
    vocab = build_vocab(filepath=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        #default='/usr/share/mscoco/annotations/captions_train2014.json', 
                        # default='./data/annotations/captions_train2014.json',
                        # default='./data/birds_captions/',
                        default='./data/text_c10/',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/flowers_vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
