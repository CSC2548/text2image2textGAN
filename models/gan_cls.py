import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import Concat_embed
import pdb

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 2400 # compatible with skip thought (1024)
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            # adding extra convs will give output (ngf*8) x 4 x 4
            nn.Conv2d(self.ngf*8, self.ngf*2, 1, 1, 0),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.Conv2d(self.ngf*2, self.ngf*2, 3, 1, 1),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.Conv2d(self.ngf*2, self.ngf*8, 3, 1, 1),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            # adding extra convs will give output (ngf*4) x 4 x 4
            nn.Conv2d(self.ngf*4, self.ngf, 1, 1, 0),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.Conv2d(self.ngf, self.ngf, 3, 1, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.Conv2d(self.ngf, self.ngf*4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),


            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=True),
            nn.Tanh()
             # state size. (num_channels) x 64 x 64
            )


    def forward(self, embed_vector, z):

        # embed_vector = 64 by 1024
        # projected_embed = 64 by 128 by 1 by 1
        # z = 64 by 100 by 1 by 1

        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.netG(latent_vector)

        return output

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = 2400 # compatible with skip thought (1024)
        self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16

        self.netD_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*8, self.ndf*2, 1, 1, 0),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf*2, self.ndf*2, 3, 1, 1),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf*2, self.ndf*8, 3, 1, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, True)

            #output size (ndf*8) x 8 x 8
        )

        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        self.netD_2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, self.ndf*8, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, inp, embed):
        x_intermediate = self.netD_1(inp)
        x = self.projector(x_intermediate, embed)
        x = self.netD_2(x)

        return x.view(-1, 1).squeeze(1) , x_intermediate


class generator2(nn.Module):
    def __init__(self):
        super(generator2, self).__init__()
        self.image_size = 128
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 2400 # compatible with skip thought (1024)
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


        # downsample
        self.netG_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ngf, 3, 1, 1),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.Conv2d(self.ngf, self.ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
            # state size. (ngf*4) x 16 x 16
        )

        self.join_embed = nn.Sequential(

            nn.Conv2d(self.projected_embed_dim + self.ngf*4 , self.ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )

        self.residual = self._make_layer(ResBlock, self.ngf * 4, 4)

        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(self.ngf * 4, self.ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(self.ngf * 2, self.ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(self.ngf, self.ngf // 2)
        # --> ngf // 4 x 128 x 128
        self.flatten = nn.Conv2d(self.ngf // 2 , self.ngf //4, 3, 1, 1)
        # --> 3 x 128 x 128
        self.img = nn.Sequential(
            nn.Conv2d(self.ngf // 4 , 3, 3, 1, 1),
            nn.Tanh()
        )

    def _make_layer(self, block, channel_num, r_num):
        layers = []
        for i in range(r_num):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)


    def forward(self, inp, embed):

        # embed_vector = 64 by 1024
        # projected_embed = 64 by 128 by 1 by 1
        # z = 64 by 100 by 1 by 1
        g1_output = self.netG_1(inp) # shape is (ngf*4) x 16 x 16
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(16, 16, 1, 1).permute(2, 3, 0, 1)
        hidden_concat = torch.cat([g1_output, replicated_embed], 1)
        x = self.join_embed(hidden_concat)
        x = self.residual(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.flatten(x)
        output = self.img(x)

        return output

class discriminator2(nn.Module):
    def __init__(self):
        super(discriminator2, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = 2400 # compatible with skip thought (1024)
        self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16

        self.encode_img = nn.Sequential(
            # state size = 3 x 128 x 128
            nn.Conv2d(3, self.ndf, 4, 2, 1, bias=True),  # 128 * 128 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            # state size = ndf x 64 x 64
            nn.Conv2d(self.ndf , self.ndf * 2, 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 8
            nn.Conv2d(self.ndf * 8, self.ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True), # 4 * 4 * ndf * 2
            nn.Conv2d(self.ndf * 2, self.ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(self.ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
            # nn.Conv2d(self.ndf * 16, self.ndf * 32, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(self.ndf * 32),
            # nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
            # conv3x3(self.ndf * 32, self.ndf * 16),
            # nn.BatchNorm2d(self.ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
            # nn.Conv2d(self.ndf * 16, self.ndf * 8, 3, 1, 1, bias=True),
            # nn.BatchNorm2d(self.ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
        )

        self.outlogitscond = nn.Sequential(
            conv3x3(self.ndf * 8 + self.projected_embed_dim, self.ndf * 8),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

        # self.outlogits = nn.Sequential(
        #     nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=4),
        #     nn.Sigmoid()
        # )


        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

    def forward(self, inp, embed):
        x_intermediate = self.encode_img(inp)
        x = self.projector(x_intermediate, embed)
        x_cond = self.outlogitscond(x)
        # x_uncond = self.outlogits(x_intermediate)

        return x_cond.view(-1, 1).squeeze(1)
