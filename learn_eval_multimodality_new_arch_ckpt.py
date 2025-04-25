from models import Model_cnn_bc, Model_cnn_mlp, Model_Cond_Diffusion
from data_preprocessing import DataHandler, CarlaCustomDataset
# from eval_diffusion_bc_multimodality_birdview import evaluate_policy
from expert_dataset import ExpertDataset
from models_bc import Model_cnn_BC
from rl_birdview_wrapper import RlBirdviewWrapper
from carla_gym.envs import EndlessFixedSpawnEnv
from data_collect import reward_configs, terminal_configs, obs_configs
import torch.utils.data as data
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
import wandb
import gym
import git
import os
from eval_diffusion_bc_multimodality_birdview_action_analyzer import eval_policy_multimodality

class TrainerNewArch():
    def __init__(self, n_epoch, lrate, device, n_hidden, batch_size, n_T,
                 net_type, drop_prob, extra_diffusion_steps, embed_dim,
                 guide_w, betas, dataset_path, run_wandb, record_run,
                 expert_dataset, name='', param_search=False,
                 embedding="Model_cnn_mlp", alpha_schedule='cosine', lrate_type='cosine',
                 start_epoch=290, model_ckpt_path=None):

        self.n_epoch = n_epoch
        self.start_epoch = start_epoch
        self.model_ckpt_path = model_ckpt_path
        self.lrate = lrate
        self.device = device
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.n_T = n_T
        self.net_type = net_type
        self.drop_prob = drop_prob
        self.extra_diffusion_steps = extra_diffusion_steps
        self.embed_dim = embed_dim
        self.guide_w = guide_w
        self.betas = betas
        self.dataset_path = dataset_path
        self.name = name
        self.param_search = param_search
        self.run_wandb = run_wandb
        self.record_rum = record_run
        self.embedding = embedding
        self.best_reward = float('-inf')
        self.patience = 20
        self.early_stopping_counter = 0
        self.expert_dataset = expert_dataset
        self.alpha_schedule = alpha_schedule
        self.lrate_type = lrate_type

    def config_wandb(self, project_name, name):
        wandb.login(key='9bcc371f01af2fc8ddab2c3ad226caad57dc4ac5')
        config={
                "n_epoch": self.n_epoch,
                "lrate": self.lrate,
                "device": self.device,
                "n_hidden": self.n_hidden,
                "batch_size": self.batch_size,
                "n_T": self.n_T,
                "net_type": self.net_type,
                "drop_prob": self.drop_prob,
                "extra_diffusion_steps": self.extra_diffusion_steps,
                "embed_dim": self.embed_dim,
                "guide_w": self.guide_w,
                "dataset": self.dataset_path,
                "model": self.embedding,
                "Architecture": self.__class__.__name__,
                "File": os.path.relpath(__file__),
                "Alpha Schedule": self.alpha_schedule,
                "commit_hash": self.get_git_commit_hash(),
                "lrate_type": self.lrate_type
            }
        if name != '':
            return wandb.init(project=project_name, name=name, config=config)
        return wandb.init(project=project_name, config=config)

    def get_git_commit_hash(self):
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha

    def prepare_dataset(self, dataset):
        obs = DataHandler().preprocess_images(dataset, observation_type='birdview')
        # obs = cv2.resize(obs[0], dsize=(96, 96), interpolation=cv2.INTER_CUBIC)[:,:,0], cmap=plt.get_cmap("gray")
        state = np.array([np.array(ele[0]['state']) for ele in dataset])
        actions = np.array([np.array(ele[0]['actions']) for ele in dataset])
        dataset = CarlaCustomDataset(obs, actions)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        '''
        The datasets have keys with the following information: birdview,
        central_rgb, left_rgb, right_rgb, item_idx, done, action, state
        '''
        return dataloader
    
    def get_x_and_y_dim(self, dataset):
        '''
        '''
        y_dim = tuple(next(iter(dataset))[1].shape)[-1]
        x_dim = tuple(next(iter(dataset))[0].shape)[1:]
        return x_dim, y_dim
    
    def create_conv_model(self, x_dim, y_dim):
        cnn_out_dim = 2
        if self.embedding == "Model_cnn_BC":
            return Model_cnn_BC(x_dim, self.n_hidden, cnn_out_dim).to(self.device)
        elif self.embedding == "Model_cnn_mlp":
            return Model_cnn_mlp(x_dim, self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type,
                                cnn_out_dim=cnn_out_dim, new_architecture=True).to(self.device)
        else:
            raise NotImplementedError
    
    def create_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lrate)
    
    def create_agent_model(self, conv_model, x_dim, y_dim):
        return Model_Cond_Diffusion(
            conv_model,
            betas=self.betas,
            n_T = self.n_T,
            device=self.device,
            x_dim=x_dim,
            y_dim=y_dim,
            drop_prob=self.drop_prob,
            guide_w=self.guide_w
        ).to(self.device)

    def loss_func(self, y, y_hat):
        return torch.nn.MSELoss()(y, y_hat)

    def decay_lr(self, epoch, lrate):
        return lrate * ((np.cos((epoch / self.n_epoch) * np.pi) + 1) / 2)

    def main(self):
        if self.run_wandb:
            self.config_wandb(project_name="Carla-Diffuser-Multimodality-New-Arch", name=self.name)

        dataload_train = self.prepare_dataset(self.expert_dataset)
        x_dim, y_dim = self.get_x_and_y_dim(dataload_train)
        model = self.create_conv_model(x_dim, y_dim)

        if self.model_ckpt_path:
            model.load_state_dict(torch.load(self.model_ckpt_path, map_location=self.device))
            print(f"Loaded model from: {self.model_ckpt_path}")

        optim = self.create_optimizer(model)
        model = self.train(model, dataload_train, optim)

    def train(self, model, dataload_train, optim):
        for ep in tqdm(range(self.start_epoch, self.n_epoch), desc="Epoch"):
            model.train()
            lr_decay = self.decay_lr(ep, self.lrate) if self.lrate_type == 'cosine' else self.lrate

            pbar = tqdm(dataload_train)
            loss_ep, n_batch = 0, 0

            for x_batch, y_batch in pbar:
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                y_hat = model(x_batch)
                loss = self.loss_func(y_hat, y_batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_ep += loss.item()
                n_batch += 1
                pbar.set_description(f"Train Loss: {loss_ep/n_batch:.4f}")

            if ep % 50 == 0 or ep in [300, 350, 400, 450, 500, 550, 600]:
                self.save_model(model, ep)

        if self.run_wandb:
            wandb.finish()
        return model

    def save_model(self, model, ep):
        os.makedirs(os.getcwd()+'/model_pytorch/' + self.model_ckpt_path.split('/')[1] +'/'+self.name, exist_ok=True)
        model_name = self.name+'_'+self.get_git_commit_hash()[0:4]+'_ep_'+f'{ep}'+'.pkl'
        torch.save(model.state_dict(), os.getcwd()+'/model_pytorch/' + self.model_ckpt_path.split('/')[1] + '/' + model_name)
        return model_name

import os
import torch
import wandb
import subprocess
from tqdm import tqdm

class TrainerNewArch_Diffusion_BC:
    def __init__(self, n_epoch, lrate, device, n_hidden, batch_size, n_T,
                 net_type, drop_prob, extra_diffusion_steps, embed_dim,
                 guide_w, betas, dataset_path, run_wandb, record_run,
                 expert_dataset, name='', param_search=False,
                 embedding="Model_cnn_mlp", alpha_schedule='cosine', lrate_type='cosine',
                 start_epoch=290, model_ckpt_path=None):

        self.n_epoch = n_epoch
        self.start_epoch = start_epoch
        self.model_ckpt_path = model_ckpt_path
        self.lrate = lrate
        self.device = device
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.n_T = n_T
        self.net_type = net_type
        self.drop_prob = drop_prob
        self.extra_diffusion_steps = extra_diffusion_steps
        self.embed_dim = embed_dim
        self.guide_w = guide_w
        self.betas = betas
        self.dataset_path = dataset_path
        self.name = name
        self.param_search = param_search
        self.run_wandb = run_wandb
        self.record_rum = record_run
        self.embedding = embedding
        self.best_reward = float('-inf')
        self.patience = 20
        self.early_stopping_counter = 0
        self.expert_dataset = expert_dataset
        self.alpha_schedule = alpha_schedule
        self.lrate_type = lrate_type

    def config_wandb(self, project_name, name):
        wandb.login(key='9bcc371f01af2fc8ddab2c3ad226caad57dc4ac5')
        config={
                "n_epoch": self.n_epoch,
                "lrate": self.lrate,
                "device": self.device,
                "n_hidden": self.n_hidden,
                "batch_size": self.batch_size,
                "n_T": self.n_T,
                "net_type": self.net_type,
                "drop_prob": self.drop_prob,
                "extra_diffusion_steps": self.extra_diffusion_steps,
                "embed_dim": self.embed_dim,
                "guide_w": self.guide_w,
                "dataset": self.dataset_path,
                "model": self.embedding,
                "Architecture": self.__class__.__name__,
                "File": os.path.relpath(__file__),
                "Alpha Schedule": self.alpha_schedule,
                "commit_hash": self.get_git_commit_hash(),
                "lrate_type": self.lrate_type
            }
        if name != '':
            return wandb.init(project=project_name, name=name, config=config)
        return wandb.init(project=project_name, config=config)

    def get_git_commit_hash(self):
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha

    def prepare_dataset(self, dataset):
        obs = DataHandler().preprocess_images(dataset, observation_type='birdview')
        # obs = cv2.resize(obs[0], dsize=(96, 96), interpolation=cv2.INTER_CUBIC)[:,:,0], cmap=plt.get_cmap("gray")
        state = np.array([np.array(ele[0]['state']) for ele in dataset])
        actions = np.array([np.array(ele[0]['actions']) for ele in dataset])
        dataset = CarlaCustomDataset(obs, actions)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        '''
        The datasets have keys with the following information: birdview,
        central_rgb, left_rgb, right_rgb, item_idx, done, action, state
        '''
        return dataloader
    
    def get_x_and_y_dim(self, dataset):
        '''
        '''
        y_dim = tuple(next(iter(dataset))[1].shape)[-1]
        x_dim = tuple(next(iter(dataset))[0].shape)[1:]
        return x_dim, y_dim
    
    def create_conv_model(self, x_dim, y_dim):
        cnn_out_dim = 4608
        if self.embedding == "Model_cnn_BC":
            return Model_cnn_BC(x_dim, self.n_hidden, cnn_out_dim).to(self.device)
        elif self.embedding == "Model_cnn_mlp":
            return Model_cnn_mlp(x_dim, self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type,
                                cnn_out_dim=cnn_out_dim, new_architecture=True).to(self.device)
        else:
            raise NotImplementedError
    
    def create_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lrate)
    
    def create_agent_model(self, conv_model, x_dim, y_dim):
        return Model_Cond_Diffusion(
            conv_model,
            betas=self.betas,
            n_T = self.n_T,
            device=self.device,
            x_dim=x_dim,
            y_dim=y_dim,
            drop_prob=self.drop_prob,
            guide_w=self.guide_w
        ).to(self.device)

    def loss_func(self, y, y_hat):
        return torch.nn.MSELoss()(y, y_hat)

    def decay_lr(self, epoch, lrate):
        return lrate * ((np.cos((epoch / self.n_epoch) * np.pi) + 1) / 2)

    def main(self):
        if self.run_wandb:
            self.config_wandb(project_name="Carla-Diffuser-Multimodality-New-Arch", name=self.name)

        dataload_train = self.prepare_dataset(self.expert_dataset)
        x_dim, y_dim = self.get_x_and_y_dim(dataload_train)
        conv_model = self.create_conv_model(x_dim, y_dim)
        model = self.create_agent_model(conv_model, x_dim, y_dim)

        if self.model_ckpt_path:
            model.load_state_dict(torch.load(self.model_ckpt_path, map_location=self.device))
            print(f"Loaded model from: {self.model_ckpt_path}")

        optim = self.create_optimizer(model)
        model = self.train(model, dataload_train, optim)

    def train(self, model, dataload_train, optim):
        distance_traveled = 0
        best_models_df = pd.DataFrame(columns=['video_path', 'model_path', 'distance_score'])
        for ep in tqdm(range(self.start_epoch, self.n_epoch), desc="Epoch"):
            # if ep > 150:
            #     break
            results_ep = [ep]
            model.train()
            if self.lrate_type == 'cosine':
                lr_decay = self.decay_lr(ep, self.lrate)
            else:
                lr_decay = self.lrate
            # train loop
            pbar = tqdm(dataload_train)
            loss_ep, n_batch = 0, 0
            if self.alpha_schedule == 'cosine':
                alpha = cosine_decay(ep, self.n_epoch)
            elif self.alpha_schedule == 'exponential':
                alpha = exponential_decay(ep, self.n_epoch)
            else:
                alpha = float(self.alpha_schedule.split('fixed_')[-1].replace('-', '.'))

            for x_batch, y_batch in pbar:
                x_batch = x_batch.type(torch.FloatTensor).to(self.device)
                y_batch = y_batch.type(torch.FloatTensor).to(self.device)
                loss_diffusion, loss_std = model.loss_on_batch(x_batch, y_batch)
                loss = alpha * loss_std + (1-alpha) * loss_diffusion
                optim.zero_grad()
                loss.backward()
                loss_ep += loss.detach().item()
                n_batch += 1
                pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
                optim.step()

                if self.run_wandb:
                    # log metrics to wandb
                    with torch.no_grad():
                        y_hat_batch = model.sample(x_batch)
                        action_MSE = extract_action_mse(y_batch, y_hat_batch)
                    wandb.log({"loss": loss_ep/n_batch,
                            "loss_diffusion": loss_diffusion,
                            "loss_std": loss_std,
                            "lr": lr_decay,
                            "alpha": alpha,
                            "steering_MSE": action_MSE[0],
                            "acceleration_MSE": action_MSE[1],
                            "distance_traveled": distance_traveled})
                        
            results_ep.append(loss_ep / n_batch)
            
            if ep % 50 == 0 or ep in [300, 350, 400, 450, 500, 550, 600]:
                self.save_model(model, ep)

    # def train(self, model, dataload_train, optim):
    #     for ep in tqdm(range(self.start_epoch, self.n_epoch), desc="Epoch"):
    #         model.train()
    #         lr_decay = self.decay_lr(ep, self.lrate) if self.lrate_type == 'cosine' else self.lrate

    #         pbar = tqdm(dataload_train)
    #         loss_ep, n_batch = 0, 0

    #         for x_batch, y_batch in pbar:
    #             x_batch = x_batch.float().to(self.device)
    #             y_batch = y_batch.float().to(self.device)
    #             y_hat = model(x_batch)
    #             loss = self.loss_func(y_hat, y_batch)
    #             optim.zero_grad()
    #             loss.backward()
    #             optim.step()
    #             loss_ep += loss.item()
    #             n_batch += 1
    #             pbar.set_description(f"Train Loss: {loss_ep/n_batch:.4f}")

    #         if ep % 50 == 0 or ep in [300, 350, 400, 450, 500, 550, 600]:
    #             self.save_model(model, ep)

    #     if self.run_wandb:
    #         wandb.finish()
    #     return model

    def save_model(self, model, ep):
        os.makedirs(os.getcwd()+'/model_pytorch/' + self.model_ckpt_path.split('/')[1] + '/' + self.model_ckpt_path.split('/')[2] + '/' + self.name, exist_ok=True)
        model_name = self.name+'_'+self.get_git_commit_hash()[0:4]+'_ep_'+f'{ep}'+'.pkl'
        torch.save(model.state_dict(), os.getcwd()+'/model_pytorch/' + self.model_ckpt_path.split('/')[1] + '/' + self.model_ckpt_path.split('/')[2] + '/' + model_name)
        return model_name

def extract_action_mse(y, y_hat):
    assert len(y) == len(y_hat)
    y_diff_pow_2 = torch.pow(y - y_hat, 2)
    y_diff_sum = torch.sum(y_diff_pow_2, dim=0)/len(y)
    mse = torch.pow(y_diff_sum, 0.5)
    return mse

def decay_lr(epoch, lrate, episodes):
    return lrate * ((np.cos((epoch / episodes) * np.pi) + 1) / 2)

def exponential_decay(epoch, total_episodes, initial_value=1.0, final_value=0.01):
    decay_rate = -np.log(final_value / initial_value) / total_episodes
    return initial_value * np.exp(-decay_rate * epoch)

def cosine_decay(epoch, total_episodes, initial_value=1.0, final_value=0.0):
    cosine_decay_value = 0.5 * (1 + np.cos(np.pi * epoch / total_episodes))
    return final_value - (final_value - initial_value) * cosine_decay_value



import numpy as np
import torch
import time
import os
import re
import math
import pandas as pd
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from carla_gym.envs import EndlessEnv, EndlessFixedSpawnEnv, LeaderboardEnv
from carla_gym.envs import EndlessFixedSpawnEnv
from models import Model_cnn_mlp, Model_Cond_Diffusion, Model_cnn_mlp_resnet, Model_cnn_mlp_original
from data_collect import reward_configs, terminal_configs, obs_configs
from data_preprocessing import DataHandler, FrontCameraMovieMakerArray
from models_bc import Model_cnn_BC
import matplotlib.pyplot as plt
from carla_route_plotter import CarlaRoutePlotter


env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0',
    # 'routes_group': 'eval'
}


spawn_point = {
    'pitch':360.0,
    'roll':0.0,
    'x':150.6903991699219,
    'y':194.78451538085938,
    'yaw':179.83230590820312,
    'z':0.0
}


spawn_point_action_histogram = {
    'pitch':360.0,
    'roll':0.0,
    'x':110.6903991699219,
    'y':194.78451538085938,
    'yaw':179.83230590820312,
    'z':0.0
}

def get_models_pathes():
    indices = {
        0: [500, 520, 600, 640, 680, 20, 30, 40, 50, 60, 70, 80, 100, 110, 120, 150, 160, 170, 190, 270, 310, 330, 340],
        1: [80, 100, 120, 130, 140, 220, 300, 480],
        2: [30, 60, 70, 80, 90, 100, 130, 140, 200, 230, 250, 340, 380, 600],
        3: [40, 130, 140, 150, 170, 180, 520, 540],
        4: [580, 500, 480, 460, 440, 420, 400, 380, 330, 280, 260, 240, 230, 210, 200, 180, 170, 160, 140, 100, 90, 80, 70, 60, 50, 40]
    }

    models = []
    base_path = "model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_"

    for version, idx_list in indices.items():
        for idx in idx_list:
            if version == 0:
                if 380 <= idx <= 740:
                    template = "new_arch_26b7_ep"
                else:
                    template = "new_arch_bc03_ep"
            elif version == 1:
                template = "new_arch_26b7_ep"
            elif version == 2:
                if 0 <= idx <= 180:
                    template = "new_arch_26b7_ep"
                elif 200 <= idx <= 280:
                    template = "new_arch_cfc7_ep"
                else:
                    template = "new_arch_37b5_ep"
            elif version == 3:
                template = "new_arch_37b5_ep"
            elif version == 4:
                template = "new_arch_37b5_ep"
            
            model_path = f"{base_path}{version}/{template}_{idx}.pkl"
            models.append(model_path)

    # Exibir os primeiros elementos para verificação
    return models


def convert_coord_dict_to_routes(coord_dict):
        """
        Convert a dictionary of coordinates into a list of routes.
        
        Args:
            coord_dict: Dictionary with keys 'x', 'y', 'z', where each key contains
                       a list of lists representing coordinates for each route.
                       Example:
                       {
                           'x': [[x1, x2, x3], [x1, x2]],
                           'y': [[y1, y2, y3], [y1, y2]],
                           'z': [[z1, z2, z3], [z1, z2]]
                       }
        
        Returns:
            List of routes, where each route is a list of (x, y, z) tuples.
            Example:
            [
                [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)],
                [(x1, y1, z1), (x2, y2, z2)]
            ]
        
        Raises:
            ValueError: If the input dictionary is missing required keys or if the
                       coordinate lists have inconsistent lengths.
        """
        # Validate input
        required_keys = {'x', 'y', 'z'}
        if not all(key in coord_dict for key in required_keys):
            missing_keys = required_keys - set(coord_dict.keys())
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        # Get number of routes and validate consistency
        n_routes = len(coord_dict['x'])
        if not all(len(coord_dict[key]) == n_routes for key in required_keys):
            raise ValueError("Inconsistent number of routes across coordinates")
        
        # Convert to list of route tuples
        routes = []
        for route_idx in range(n_routes):
            # Validate route point consistency
            route_lengths = [len(coord_dict[key][route_idx]) for key in required_keys]
            if not all(length == route_lengths[0] for length in route_lengths):
                raise ValueError(f"Inconsistent coordinate lengths in route {route_idx}")
            
            # Create route points
            route_points = []
            for point_idx in range(route_lengths[0]):
                point = (
                    coord_dict['x'][route_idx][point_idx],
                    coord_dict['y'][route_idx][point_idx],
                    coord_dict['z'][route_idx][point_idx]
                )
                route_points.append(point)
            routes.append(route_points)
        
        return routes

def handle_obs(obs, observation_type, embedding):
    obs = DataHandler().preprocess_images(obs, observation_type=observation_type , eval=True, embedding=embedding)
    return obs

def evaluate_policy(env, model, video_path, device, max_eval_steps=3000, observation_type='birdview',  architecture='diffusion', movie=True, extra_steps=0, embedding=Model_cnn_mlp, persist_points = None, plotter=None):
    
    # max_eval_steps = 10
    
    model = model.eval()
    t0 = time.time()
    obs = env.reset()
    previous_position = obs['gnss']
    obs = handle_obs(obs, observation_type, embedding=embedding)
    n_step = 0
    env_done = False
    list_render = []
    ep_stat_buffer = []
    route_completion_buffer = pd.DataFrame(columns=['step', 'simulation_time', 'route_completed_in_m', 'route_length_in_m', 'is_route_completed'])
    route_infraction = pd.DataFrame(columns=['c_blocked', 'c_lat_dist', 'c_collision', 'collision', 'c_collision_px', 'timeout', 'info_dict', 'lat_dist', 'thresh_lat_dist'])
    route_infraction_2 = pd.DataFrame(columns=['collisions_layout','collisions_vehicle','collisions_pedestrian','collisions_others','route_deviation','wrong_lane','outside_lane','run_red_light','encounter_stop','stop_infraction'])
    list_render_front = []
    list_render_left = []
    list_render_right = []
    list_render_birdview = []
    list_locations = []
    ep_dict = {}
    ep_dict['actions'] = []
    ep_dict['state'] = []
    distance_traveled = 0
    while n_step < max_eval_steps:
        if architecture == 'diffusion':
            actions = model.sample_extra(torch.tensor(obs).float().to(device), extra_steps=extra_steps).to(device)[0]
        elif architecture == 'mse':
            actions = model(torch.tensor(obs).float().to(device)).to(device)[0]
        obs_clean, reward, done, info = env.step(np.array(actions.detach().cpu()))
        if n_step == 0:
            previous_position = np.array(info['location'])
        distance_traveled += calculate_distance_traveled(np.array(info['location']), previous_position)

        new_row = pd.DataFrame([info['route_completion']])
        route_completion_buffer = pd.concat([route_completion_buffer, new_row], ignore_index=True)

        filtered_data = {key: info['terminal_debug'][key] for key in route_infraction.columns if key in info['terminal_debug']}
        filtered_data_2 = {key: info[key] for key in route_infraction_2.columns if key in info}
        new_row = pd.DataFrame([filtered_data])
        new_row_2 = pd.DataFrame([filtered_data_2])
        route_infraction = pd.concat([route_infraction, new_row], ignore_index=True)
        route_infraction_2 = pd.concat([route_infraction_2, new_row_2], ignore_index=True)

        route_infraction_total = pd.concat([route_infraction, route_infraction_2], axis=1)

        obs = handle_obs(obs_clean, observation_type, embedding)
        
        if True:
            list_render.append(np.transpose(obs_clean['central_rgb'], (1,2,0)))
            list_render_front.append(np.transpose(obs_clean['central_rgb'], (1,2,0)))
            list_render_right.append(np.transpose(obs_clean['right_rgb'], (1,2,0)))
            list_render_left.append(np.transpose(obs_clean['left_rgb'], (1,2,0)))
            list_render_birdview.append(np.transpose(obs_clean['birdview'], (1,2,0)))
            list_locations.append(info['location'])
            ep_dict['state'].append(np.transpose(obs_clean['birdview'], (1,2,0)))
            ep_dict['actions'].append([actions[0].item(), actions[1].item()])
            if info['location'][0] < 85:
                break
        else:
            list_render.append(env.render(mode='rgb_array'))
        n_step += 1
        env_done = done
        
        print(f'ep: {int(video_path[:-4].split("_")[-1])} ---- n_step: {n_step} ---- distance_traveled: {distance_traveled}')

        # for i in np.where(done)[0]:
        #     break

        if sum(list(route_infraction_total.collisions_layout.values)) > 0 or sum(list(route_infraction_total.wrong_lane.values)) > 0 or sum(list(route_infraction_total.route_deviation.values)) > 0:
            break
        # if n_step > 2:
        #     break

    if True:
        # if movie:
        #     movie_maker = FrontCameraMovieMakerArray(video_path=video_path,
        #                                             front_array=list_render_front,
        #                                             left_array=list_render_left,
        #                                             right_array=list_render_right,
        #                                             birdview_array=list_render_birdview)
        #     movie_maker.save_record(text=f'{distance_traveled}')
        # plotter.plot_routes_from_list(persist_points_list, 'paper_plots/traj_plot')
        # persist_points = plot_gnss_2d(list_locations, output_path=video_path[:-6]+".png", persist_points=persist_points)
        persist_points = plot_gnss_2d(list_locations, output_path=video_path[:-4]+".png", persist_points=persist_points)
        persist_points_list = convert_coord_dict_to_routes(persist_points)
        plotter.plot_routes_from_list(persist_points_list, video_path[:-4]+"_map.png")
        # plot_left_right_trajectories(output_path=video_path[:-6]+"_trajectory.png", persist_points=persist_points)
        # plot_left_right_trajectories(output_path=video_path[:-6]+"_trajectory.png", persist_points=persist_points)
        gnss_path = video_path[:-3]+'txt'
        route_completion_data_path = video_path[:-3]+'csv'
        actions_observation_path = video_path[:-3]+'json'

        route_info = pd.concat([route_completion_buffer, route_infraction_total], axis=1)
        route_info.to_csv(route_completion_data_path, index=False)

        np.savetxt(video_path[:-4]+'_distance_traveled.txt', [distance_traveled])
        np.savetxt(gnss_path, list_locations)
        ep_df = pd.DataFrame(ep_dict)
        # ep_df.to_json(actions_observation_path)
    return distance_traveled, None

def plot_left_right_trajectories(output_path, persist_points):
    classifications = {'Left': 0, 'Straight': 0, 'Right': 0}
    
    for sublist in persist_points['x']:
        if not sublist:
            continue  # Pula listas vazias
        last_value = sublist[-1]
        
        if last_value < 180:
            classifications['Left'] += 1
        elif last_value > 210:
            classifications['Right'] += 1
        else:
            classifications['Straight'] += 1
    
    # Plotando histograma
    plt.bar(classifications.keys(), classifications.values(), color=['red', 'blue', 'green'])
    plt.xlabel('Classes')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Classificações')
    plt.show()
    plt.savefig(output_path)
    plt.close()

def plot_gnss_2d(list_gnss, output_path="gnss_plot.png", persist_points=None):
    if persist_points is None:
        persist_points = {"x": [], "y": [], "z": []}

    # Adicionar novos pontos ao histórico
    x_coords = [array[0] for array in list_gnss]
    y_coords = [array[1] for array in list_gnss]
    z_coords = [array[2] for array in list_gnss]
    persist_points["x"].append(x_coords)
    persist_points["y"].append(y_coords)
    persist_points["z"].append(z_coords)
    
    # Criar ou atualizar o gráfico
    plt.figure(figsize=(8, 6))
    for x_seq, y_seq in zip(persist_points["x"], persist_points["y"]):
        plt.plot(x_seq, y_seq, marker='o')

    plt.title("2D GNSS Coordinates (x, y) - Updated")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # plt.xlim([-0.0018, -0.0017])
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    # Garantir que o diretório para salvar o arquivo exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar o gráfico como PNG
    plt.savefig(output_path)
    plt.close()

    # Retornar as coordenadas persistidas
    return persist_points



def update_dataframe(df, route_completion):
    df = df.append(route_completion, ignore_index=True)

def env_maker(multimodality):
    if multimodality:
        return env_maker_multimodality
    else:
        return env_maker_fixed_route

def env_maker_fixed_route():

    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2001,
                    seed=100, 
                    no_rendering=True, **env_configs)
    env = RlBirdviewWrapper(env)
    return env

def env_maker_multimodality():

    # env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
    #                 terminal_configs=terminal_configs, host='localhost', port=3001,
    #                 seed=np.random.randint(1, 3001), 
    #                 no_rendering=True, **env_configs)

    env_configs = {
        'carla_map': 'Town01',
        'num_zombie_vehicles': [0, 150],
        'num_zombie_walkers': [0, 300],
        'weather_group': 'dynamic_1.0'
        }

    # env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
    #                 terminal_configs=terminal_configs, host='localhost', port=2001,
    #                 seed=np.random.randint(1, 3001), 
    #                 no_rendering=True, **env_configs, spawn_point=spawn_point)

    env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2001,
                    seed=np.random.randint(1, 3001), 
                    no_rendering=True, **env_configs)
    env = RlBirdviewWrapper(env)
    return env


def calculate_distance_traveled(current_position, previous_position):
    """
    Calcula a distância percorrida por um veículo desde a última posição registrada.
    
    Args:
        vehicle: Objeto do veículo no CARLA.
        previous_position: Última posição registrada do veículo (carla.Location).
    
    Returns:
        distance: Distância percorrida desde a última posição (float).
        current_position: Posição atual do veículo (carla.Location).
    """
    
    if previous_position is None:
        # Primeira chamada: sem deslocamento
        return 0.0, current_position

    # Calcular o deslocamento 3D
    dx = current_position[0] - previous_position[0]
    dy = current_position[1] - previous_position[1]
    dz = current_position[2] - previous_position[2]
    distance = math.sqrt(dx**2 + dy**2)
    
    return distance


def list_pkl_files_sorted(directory):
    pkl_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pkl')]
    return sorted(pkl_files)



def encontrar_arquivos_pkl(diretorio):
    arquivos_pkl = []

    for raiz, diretorios, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.endswith(".pkl"):
                caminho_completo = os.path.join(raiz, arquivo)
                arquivos_pkl.append((os.path.getctime(caminho_completo), caminho_completo))

    arquivos_pkl.sort()

    return [caminho for _, caminho in arquivos_pkl[::-1]]


def sort_by_filename_and_version(file_paths):
    def get_sort_key(path):
        # Extract the filename
        filename = os.path.basename(path)
        
        # Extract the epoch number from the filename
        # Looking for pattern like 'new_arch_26b7_ep_10.pkl'
        epoch_match = re.search(r'ep_(\d+)\.pkl', filename)
        epoch_num = int(epoch_match.group(1)) if epoch_match else 0
        
        # Extract the version number from the folder path
        # Looking for pattern like 'version_750_2'
        version_match = re.search(r'version_\d+_(\d+)', path)
        version_num = int(version_match.group(1)) if version_match else 0
        
        # Return a tuple that will sort first by epoch, then by version
        return (epoch_num, version_num)
    
    return sorted(file_paths, key=get_sort_key)


if __name__ == '__main__':

    trainer = TrainerNewArch(
        n_epoch=701,
        start_epoch=0,
        lrate=1e-4,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        n_hidden=128,
        batch_size=32,
        n_T=20,
        net_type="mse",
        drop_prob=0.0,
        extra_diffusion_steps=0,
        embed_dim=64,
        guide_w=1.0,
        betas=(1e-4, 0.02),
        dataset_path='data_collection/town01_multimodality_t_insersection_multiples_trajectory',
        run_wandb=False,
        record_run=True,
        expert_dataset=ExpertDataset('data_collection/town01_multimodality_t_insersection_multiples_trajectory'),
        name="version_750",
        embedding="Model_cnn_BC",
        model_ckpt_path=None
    )
    trainer.main()



    trainer = TrainerNewArch_Diffusion_BC(
        n_epoch=701,
        start_epoch=0,
        lrate=1e-4,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        n_hidden=128,
        batch_size=32,
        n_T=20,
        net_type="transformer",
        drop_prob=0.0,
        extra_diffusion_steps=0,
        embed_dim=128,
        guide_w=0.0,
        betas=(1e-4, 0.02),
        dataset_path='data_collection/town01_multimodality_t_insersection_multiples_trajectory',
        run_wandb=False,
        record_run=True,
        expert_dataset=ExpertDataset('data_collection/town01_multimodality_t_insersection_multiples_trajectory'),
        name="version_750",
        embedding="Model_cnn_mlp",
        model_ckpt_path=None,
        alpha_schedule='fixed_0-0',
        lrate_type='cosine'
    )
    trainer.main()


    trainer = TrainerNewArch_Diffusion_BC(
        n_epoch=701,
        start_epoch=600,
        lrate=1e-4,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        n_hidden=128,
        batch_size=32,
        n_T=20,
        net_type="transformer",
        drop_prob=0.0,
        extra_diffusion_steps=0,
        embed_dim=128,
        guide_w=0.0,
        betas=(1e-4, 0.02),
        dataset_path='data_collection/town01_multimodality_t_insersection_multiples_trajectory',
        run_wandb=False,
        record_run=True,
        expert_dataset=ExpertDataset('data_collection/town01_multimodality_t_insersection_multiples_trajectory'),
        name="new_arch_fixed",
        embedding="Model_cnn_mlp",
        model_ckpt_path=None,
        alpha_schedule='fixed_0-3',
        lrate_type='cosine'
    )
    trainer.main()

# # ----------------------------------------------------------------------------------

#     env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
#                 terminal_configs=terminal_configs, host='localhost', port=2020,
#                 seed=np.random.randint(1, 3001), no_rendering=True, **env_configs)
    
#     env = RlBirdviewWrapper(env)

#     models_0 = encontrar_arquivos_pkl('model_pytorch/BC_Full_Trajectory_300_00')
#     models_1 = encontrar_arquivos_pkl('model_pytorch/BC_Full_Trajectory_300_01')
#     models_2 = encontrar_arquivos_pkl('model_pytorch/BC_Full_Trajectory_300_02')

#     models = models_2 + models_0 + models_1

#     models = sort_by_filename_and_version(models)

#     device = 'cuda'
#     x_shape = (192, 192, 4)
#     y_dim = 2
#     embed_dim = 64
#     n_hidden = 128

#     model = Model_cnn_BC(x_shape=(192, 192, 4), n_hidden=128, cnn_out_dim=2).to(device)


#     # -----------------------------------------------------------------------------------------
#     extra_steps_list = [0]
#     plotter = CarlaRoutePlotter(host='localhost', port=2030, town='Town01')
#     # for extra_steps in extra_steps_list:
#     #     for model_path in models:
#     #         if int(model_path.split('.')[0].split('_')[-1]) < 300:
#     #             continue
#     #         if int(model_path.split('.')[0].split('_')[-1]) % 50 != 0 and int(model_path.split('.')[0].split('_')[-1]) != 290:
#     #             continue
#     #         model.load_state_dict(torch.load(model_path))
#     #         persist_points = None
#     #         diff_bc_video = f'diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/{model_path.split("/")[1]}_{extra_steps}_extra_steps/'
#     #         diff_bc_video_2 = diff_bc_video + model_path.split('/')[-2] + '/'
#     #         os.makedirs(diff_bc_video_2, exist_ok=True)
#     #         eval_video_path = diff_bc_video_2 + model_path.split('/')[-1].split('.')[0] + f'_{0}' + '.mp4'
#     #         if os.path.exists(eval_video_path[:-4]+"_map.png"):
#     #             continue
#     #         for i in range(10):
#     #             diff_bc_video = f'diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/{model_path.split("/")[1]}_{extra_steps}_extra_steps/'
#     #             diff_bc_video_2 = diff_bc_video + model_path.split('/')[-2] + '/'
#     #             os.makedirs(diff_bc_video_2, exist_ok=True)
#     #             eval_video_path = diff_bc_video_2 + model_path.split('/')[-1].split('.')[0] + f'_{i}' + '.mp4'
#     #             _, persist_points= evaluate_policy(
#     #                                 env=env,
#     #                                 model=model.to(device),
#     #                                 video_path=eval_video_path,
#     #                                 device=device,
#     #                                 observation_type='birdview',
#     #                                 max_eval_steps=3000,
#     #                                 architecture='mse',
#     #                                 movie=True,
#     #                                 extra_steps=extra_steps,
#     #                                 embedding='Model_cnn_mlp',
#     #                                 persist_points = persist_points,
#     #                                 plotter=plotter)
                
# # Fixed DBC -------------------------------------------------------------------

#     models_0 = encontrar_arquivos_pkl('model_pytorch/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_0/version_750_0')
#     models_1 = encontrar_arquivos_pkl('model_pytorch/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_1/version_750_0')
#     models_2 = encontrar_arquivos_pkl('model_pytorch/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_2/version_750_0')
#     models_3 = encontrar_arquivos_pkl('model_pytorch/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_0/version_750_2')
#     models_4 = encontrar_arquivos_pkl('model_pytorch/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_1/version_750_2')
#     models_5 = encontrar_arquivos_pkl('model_pytorch/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_2/version_750_2')
#     models = models_0 + models_1 + models_2 + models_3 + models_4 + models_5

#     models = sort_by_filename_and_version(models)
    
#     observation_type = 'birdview'
#     device = 'cuda'
#     net_type = 'transformer'

#     x_shape = (192, 192, 4)
#     y_dim = 2
#     embed_dim = 128
#     n_hidden = 128

#     nn_model = Model_cnn_mlp(
#         x_shape,
#         n_hidden,
#         y_dim,
#         embed_dim=embed_dim,
#         net_type=net_type,
#         cnn_out_dim=4608).to(device)

#     model = Model_Cond_Diffusion(
#         nn_model,
#         betas=(1e-4, 0.02),
#         n_T=20,
#         device=device,
#         x_dim=x_shape,
#         y_dim=2,
#         drop_prob=0.0,
#         guide_w=0.0,)

#     # -----------------------------------------------------------------------------------------
#     extra_steps_list = [0]
#     # plotter = CarlaRoutePlotter(host='localhost', port=2030, town='Town01')
#     for extra_steps in extra_steps_list:
#         for model_path in models:
#             if int(model_path.split('.')[0].split('_')[-1]) < 300:
#                 continue
#             if int(model_path.split('.')[0].split('_')[-1]) % 50 != 0 and int(model_path.split('.')[0].split('_')[-1]) != 290:
#                 continue
#             model.load_state_dict(torch.load(model_path))
#             persist_points = None
#             diff_bc_video = f'diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC/{model_path.split("/")[1]}_{extra_steps}_extra_steps/'
#             diff_bc_video_2 = diff_bc_video + model_path.split('/')[-2] + '/'
#             os.makedirs(diff_bc_video_2, exist_ok=True)
#             eval_video_path = diff_bc_video_2 + model_path.split('/')[-1].split('.')[0] + f'_{0}' + '.mp4'
#             if os.path.exists(eval_video_path[:-4]+"_map.png"):
#                 continue
#             for i in range(10):
#                 diff_bc_video = f'diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC/{model_path.split("/")[1]}_{extra_steps}_extra_steps/'
#                 diff_bc_video_2 = diff_bc_video + model_path.split('/')[-2] + '/'
#                 os.makedirs(diff_bc_video_2, exist_ok=True)
#                 eval_video_path = diff_bc_video_2 + model_path.split('/')[-1].split('.')[0] + f'_{i}' + '.mp4'
#                 _, persist_points= evaluate_policy(
#                                     env=env,
#                                     model=model.to(device),
#                                     video_path=eval_video_path,
#                                     device=device,
#                                     observation_type=observation_type,
#                                     max_eval_steps=3000,
#                                     architecture='diffusion',
#                                     movie=True,
#                                     extra_steps=extra_steps,
#                                     embedding='Model_cnn_mlp',
#                                     persist_points = persist_points,
#                                     plotter=plotter)