import torch

obs = torch.load("/home/casa/projects/bruno/carla-diffuser-bc/bet_data_release/carla/all_observations.pth")
actions = torch.load("/home/casa/projects/bruno/carla-diffuser-bc/bet_data_release/carla/all_actions_pm1.pth")
seq = torch.load("/home/casa/projects/bruno/carla-diffuser-bc/bet_data_release/carla/seq_lengths.pth")
stop = True