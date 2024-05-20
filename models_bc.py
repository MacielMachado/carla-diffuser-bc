import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from models import ResidualConvBlock



class Model_cnn_BC(nn.Module):
    def __init__(self, x_shape, n_hidden, cnn_out_dim=2):
        super(Model_cnn_BC, self).__init__()

        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.n_feat = 64

        # set up CNN for image
        self.conv_down1 = nn.Sequential(
            ResidualConvBlock(self.x_shape[-1], self.n_feat, is_res=True),
            nn.MaxPool2d(2),
        )
        self.conv_down3 = nn.Sequential(
            ResidualConvBlock(self.n_feat, self.n_feat * 2, is_res=True),
            nn.MaxPool2d(2),
        )
        self.imageembed = nn.Sequential(nn.AvgPool2d(8))

        # cnn_out_dim = self.n_feat * 2  # how many features after flattening -- WARNING, will have to adjust this for diff size input resolution
        cnn_out_dim = cnn_out_dim
        # it is the flattened size after CNN layers, and average pooling

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1)  # [batch_size, 128, 35, 18]
        # c3 is [batch size, 128, 4, 4]
        x_embed = self.imageembed(x3)
        # c_embed is [batch size, 128, 1, 1]
        x_embed = x_embed.view(x.shape[0], -1)
        # c_embed is now [batch size, 128]
        return x_embed