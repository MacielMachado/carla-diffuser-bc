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


class Model_cnn_GKC(nn.Module):
    def __init__(self, x_shape, y_dim, embed_dim, net_type, observation_space=None, features_dim=256, states_neurons=[256], output_dim=None, cnn_out_dim=1152, embed_n_hidden=128):
        super(Model_cnn_GKC, self).__init__()

        self.x_shape = x_shape
        self.x_shape = (192, 192, 3)
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.n_feat = 64
        self.net_type = net_type
        self.embed_n_hidden = embed_n_hidden


        self.cnn = nn.Sequential(
            nn.Conv2d(self.x_shape[-1], 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            example = np.random.rand(*self.x_shape)
            n_flatten = self.cnn(torch.as_tensor(np.expand_dims(np.transpose(example, (2,0,1)), axis=0)).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+states_neurons[-1], 512), nn.ReLU(),
                                    nn.Linear(512, features_dim), nn.ReLU())

        states_neurons = [y_dim*2] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

        self.head = nn.Sequential(nn.Linear(features_dim, features_dim), nn.ReLU(),
                                  nn.Linear(features_dim, features_dim), nn.ReLU(),
                                  nn.Linear(features_dim, 2), nn.ReLU(),)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x, speed, previous_action):
        x_embed = self.embed_context(x, speed, previous_action)

        return x_embed

    def embed_context(self, x, speed, previous_action):
        x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(192, 192), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        x = x.permute(0, 3, 2, 1)
   
        x = self.cnn(x)
        latent_state = self.state_linear(torch.cat((speed, previous_action), dim=1))

        x = torch.cat((x, latent_state), dim=1)
        x = self.linear(x)
        x_embed = self.head(x)
        return x_embed