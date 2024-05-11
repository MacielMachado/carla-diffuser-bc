# Image Folder
# Scheduler
# Transfer Learning

from expert_dataset import ExpertDataset
from data_preprocessing import DataHandler, CarlaCustomDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from skimage.transform import resize

dataset = ExpertDataset('gail_experts_multi_bruno_3_simples', n_routes=2, n_eps=10)
obs = DataHandler().preprocess_images(dataset, feature='front')
actions = np.array([np.array(ele[0]['actions']) for ele in dataset])
dataset = CarlaCustomDataset(obs, actions)
print("4.4")
dataloader = data.DataLoader(dataset,
                                batch_size=32,
                                shuffle=True)
device = 'cpu'
model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 4096)

first_conv_layer = [nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
first_conv_layer.extend(list(model.children()))  
model = nn.Sequential(*first_conv_layer)  

model = model.to(device)

obs_instance = next(iter(dataloader))[0].permute(0, 3, 2, 1)

obs_instance_np = np.array(obs_instance)
obs_instance_torch = torch.Tensor(resize(obs_instance_np[:,:,:,:], (obs_instance_np.shape[0], obs_instance_np.shape[1],224, 224), mode='constant')).type(torch.float32)
model(obs_instance_torch[0:1])

output = model(obs_instance.type(torch.float32))

# Primeiro teste: tamanho (1, 3, 224, 224) ✅
x = torch.randn(1, 3, 224, 224)
models.resnet18(pretrained=True)(x)

# Segundo teste: tamanho (1, 12, 224, 224) ✅
x = torch.randn(1, 12, 224, 224)
first_conv_layer = [nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
model = nn.Sequential(*first_conv_layer)  
x_2 = model(x)
models.resnet18(pretrained=True)(x_2)

# Terceiro teste: imagem ✅
obs_instance = next(iter(dataloader))[0].permute(0, 3, 2, 1)
obs_instance_np = np.array(obs_instance)
obs_instance_torch = torch.Tensor(resize(obs_instance_np[:,:,:,:], (obs_instance_np.shape[0], obs_instance_np.shape[1],224, 224), mode='constant')).type(torch.float32)[0:1]
first_conv_layer = [nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
model = nn.Sequential(*first_conv_layer)  
x_3 = model(obs_instance_torch)
models.resnet18(pretrained=True)(x_3)

# Quarto teste: imagem cheia 
obs_instance = next(iter(dataloader))[0].permute(0, 3, 2, 1)
obs_instance_np = np.array(obs_instance)
obs_instance_torch = torch.Tensor(resize(obs_instance_np[:,:,:,:], (obs_instance_np.shape[0], obs_instance_np.shape[1],224, 224), mode='constant')).type(torch.float32)[0:1]
first_conv_layer = [nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
model = models.resnet18(pretrained=True)
first_conv_layer.extend(list(model.children()))  
model = nn.Sequential(*first_conv_layer)  

# Quinto teste: 
obs_instance = next(iter(dataloader))[0].permute(0, 3, 2, 1)
obs_instance_np = np.array(obs_instance)
obs_instance_torch = torch.Tensor(resize(obs_instance_np[:,:,:,:], (obs_instance_np.shape[0], obs_instance_np.shape[1],224, 224), mode='constant')).type(torch.float32)[0:1]
model = models.resnet18(pretrained=True)
weights = model.conv1.weight
new_weights = torch.cat([weights] * 4, dim=1)
model.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.conv1.weight = torch.nn.Parameter(new_weights)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4096)

model(obs_instance_torch)