import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import *
from caption_gan_encoder_decoder_model import EncoderCNN
from caption_gan_encoder_decoder_model import DecoderRNN
import pdb

class CaptionDiscriminator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionDiscriminator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.image_feature_encoder = EncoderCNN(embed_size)
        self.sentence_feature_encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden_fine_tune_linear = nn.Linear(hidden_size, embed_size)

    def forward(self, images, captions, lengths):
        """Calculate reward score: r = logistic(dot_prod(f, h))"""
        # print(captions)
        features = self.image_feature_encoder(images) #(batch_size=128, embed_size=256)

        if torch.cuda.is_available(): 
            embeddings = self.embed(captions.type(torch.cuda.LongTensor)) # (batch_size, embed_size)
        else:
            embeddings = self.embed(captions.type(torch.LongTensor)) # (batch_size, embed_size)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.sentence_feature_encoder(packed)

        padded = pad_packed_sequence(hiddens, batch_first=True)
        # padded[0] # (batch_size, T_max, hidden_size)
        last_padded_indices = [index-1 for index in padded[1]]
        hidden_outputs = padded[0][range(captions.size(0)), last_padded_indices, :]
        hidden_outputs = self.hidden_fine_tune_linear(hidden_outputs)
        
        dot_prod = torch.bmm(features.unsqueeze(1), hidden_outputs.unsqueeze(1).transpose(2,1)).squeeze()
        return nn.Sigmoid()(dot_prod)


class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionGenerator, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        self.features = None

    def forward(self, images, captions, lengths):
        """Getting captions"""
        self.features = self.encoder(images)
        outputs, packed_lengths = self.decoder(self.features, captions, lengths, noise=False) # TODO (packed_size, vocab_size)
        # outputs = self.decoder(self.features, captions, lengths, noise=False) # TODO (packed_size, vocab_size)
        # outputs = PackedSequence(outputs, packed_lengths)
        # outputs = pad_packed_sequence(outputs, batch_first=True) # (b, T, V)
        return outputs, packed_lengths


    def pre_compute(self, gen_samples, t):
        """
            pre compute the most likely vocabs and their states
        """
        if self.features is None:
            print('must do forward before calling this function')
            return None

        predicted_ids, saved_states = self.decoder.pre_compute(self.features, gen_samples, t)
        return predicted_ids, saved_states

    def rollout(self, gen_samples, t, saved_states):
        """ inputs:
                * gen_samples: (b, Tmax)
                * t: scalar

            outputs:
                * gen_rollouts: (b, Tmax - t)
                * lengths_rollouts: list (b)
        """
        if self.features is None:
            print('must do forward before calling this function')
            return None

        Tmax = gen_samples.size(1)
        sampled_ids = self.decoder.rollout(self.features, gen_samples, t, Tmax, states=saved_states)
        # pdb.set_trace()
        return sampled_ids

    def sample(self, images, states=None):
        features = self.encoder(images)
        return self.decoder.sample(features, states)

