import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.nn.functional as F
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

        self.fc1 = nn.Linear(4608, 2304)
        self.fc2 = nn.Linear(2304, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.fc4 = nn.Linear(64, cnn_out_dim)

        # cnn_out_dim = self.n_feat * 2  # how many features after flattening -- WARNING, will have to adjust this for diff size input resolution
        # it is the flattened size after CNN layers, and average pooling

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1)  # [batch_size, 128, 35, 18]
        # c3 is [batch size, 128, 4, 4]
        x_embed = self.imageembed(x3)
        # c_embed is [batch size, 128, 1, 1]
        x_embed = x_embed.view(x.shape[0], -1)
        x_lin1 = self.fc1(x_embed)
        x_lin1 = F.relu(x_lin1)

        x_lin2 = self.fc2(x_lin1)
        x_lin2 = F.relu(x_lin2)

        x_lin3 = self.fc3(x_lin2)
        x_lin3 = F.relu(x_lin3)

        x_lin4 = self.fc4(x_lin3)
        # c_embed is now [batch size, 128]
        return x_lin4


class Model_cnn_BC_resnet(nn.Module):
    def __init__(self, x_shape, n_hidden, cnn_out_dim=2, resnet_depth="18", origin="birdview"):
        super(Model_cnn_BC_resnet, self).__init__()

        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.n_feat = 64
        self.resnet_depth = resnet_depth

        if origin == 'birdview':
            num_channels = 4
        elif origin == 'front':
            num_channels = 12
        else:
            raise NotImplementedError

        if self.resnet_depth == '18':
            self.model = models.resnet18(pretrained=True) 
        elif self.resnet_depth == '50':
            self.model = models.resnet50(pretrained=True)

        weights = self.model.conv1.weight
        if origin == 'front':
            weights = torch.cat([weights] * 4, dim=1)

        self.new_conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.num_fc_ftrs = self.model.fc.in_features
        self.new_fc = nn.Linear(self.num_fc_ftrs, cnn_out_dim)

        self.model.conv1 = self.new_conv1
        self.model.conv1.weight = torch.nn.Parameter(weights)
        self.model.fc = self.new_fc

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        return self.model(x)