import numpy as np
import torch
import time
import os
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_birdview_wrapper import RlBirdviewWrapper
# from carla_gym.envs import EndlessEnv, EndlessFixedSpawnEnv, LeaderboardEnv
from carla_gym.envs import EndlessFixedSpawnEnv
from models import Model_cnn_mlp, Model_Cond_Diffusion, Model_cnn_mlp_resnet
from data_collect import reward_configs, terminal_configs, obs_configs
from data_preprocessing import DataHandler, FrontCameraMovieMakerArray
from models_bc import Model_cnn_BC

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


def handle_obs(obs, observation_type):
    obs = DataHandler().preprocess_images(obs, observation_type=observation_type , eval=True)
    return obs

def evaluate_policy(env, model, video_path, device, max_eval_steps=3000, observation_type='birdview',  architecture='diffusion'):
    model = model.eval()
    t0 = time.time()
    # for i in range(env.num_envs):
    #     env.set_attr('eval_mode', True, indices=1)
    obs = handle_obs(env.reset(), observation_type)
    n_step = 0
    env_done = False
    list_render = []
    ep_stat_buffer = []
    route_completion_buffer = []
    list_render_front = []
    list_render_left = []
    list_render_right = []
    list_render_birdview = []
    list_gnss = []

    while n_step < max_eval_steps and env_done == False:
        if architecture == 'diffusion':
            actions = model.sample(torch.tensor(obs).float().to(device)).to(device)[0]
        elif architecture == 'mse':
            actions = model(torch.tensor(obs).float().to(device)).to(device)[0]
        obs_clean, reward, done, info = env.step(np.array(actions.detach().cpu()))
        obs = handle_obs(obs_clean, observation_type)
        
        if observation_type == 'front':
            list_render.append(np.transpose(obs_clean['central_rgb'], (1,2,0)))
            list_render_front.append(np.transpose(obs_clean['central_rgb'], (1,2,0)))
            list_render_right.append(np.transpose(obs_clean['right_rgb'], (1,2,0)))
            list_render_left.append(np.transpose(obs_clean['left_rgb'], (1,2,0)))
            list_render_birdview.append(np.transpose(obs_clean['birdview'], (1,2,0)))
            list_gnss.append(info['gnss'])
        else:
            list_render.append(env.render(mode='rgb_array'))
        n_step += 1
        env_done = done
        
        print(f'n_step: {n_step}')

        for i in np.where(done)[0]:
            break

    if observation_type == 'front':
        movie_maker = FrontCameraMovieMakerArray(video_path=video_path,
                                                front_array=list_render_front,
                                                left_array=list_render_left,
                                                right_array=list_render_right,
                                                birdview_array=list_render_birdview)
        gnss_path = video_path[:-3]+'txt'
        np.savetxt(gnss_path, list_gnss)
        movie_maker.save_record()

    if observation_type != 'front':
        encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()

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

    env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2001,
                    seed=np.random.randint(1, 3001), 
                    no_rendering=True, **env_configs, spawn_point=spawn_point)
    env = RlBirdviewWrapper(env)
    return env


if __name__ == '__main__':
    diff_bc_video = 'diff_bc_video_(not_diffuser)/multi_birdview/'
    diff_bc_video = 'diff_bc_video_(diffuser)/birdview/teste_3/'

    diff_bc_video = 'diff_bc_video_(not_diffuser)/multi_birdview/'
    os.makedirs(diff_bc_video, exist_ok=True)

    device = 'cuda'
    net_type = 'transformer'
    observation_type = 'front'

    x_shape = (192, 192, 4)
    y_dim = 2
    embed_dim = 128
    n_hidden = 128

    # nn_model = Model_cnn_mlp(
    #     x_shape,
    #     n_hidden,
    #     y_dim,
    #     embed_dim=embed_dim,
    #     net_type=net_type,
    #     cnn_out_dim=4096).to(device)
    
    x_shape=(224, 224, 12)
    # nn_model = Model_cnn_mlp_resnet(
    #     x_shape=x_shape,
    #     n_hidden=n_hidden,
    #     y_dim=y_dim,
    #     embed_dim=embed_dim,
    #     net_type=net_type,
    #     origin=observation_type,
    #     cnn_out_dim=4608,
    #     resnet_depth='18'
    # )

    # model = Model_Cond_Diffusion(
    #     nn_model,
    #     betas=(1e-4, 0.02),
    #     n_T=50,
    #     device=device,
    #     x_dim=x_shape,
    #     y_dim=2,
    #     drop_prob=0.0,
    #     guide_w=0.0,)
    
    # model_path = 'model_pytorch/multi/gail_experts_semaphores_nroutes1_neps1_c4c3_ep_749.pkl'
    models = [
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_749.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_600.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_500.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_400.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_350.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_300.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_250.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_200.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_150.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_120.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_100.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_90.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_80.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_70.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_60.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_50.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_40.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_20.pkl',
        'model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_1.pkl',]

    models = ['model_pytorch_multi_full_front_resnet18_2/gail_experts_nroutes1_neps1_f245_ep_600.pkl',]

    # models = [
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_1.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_20.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_30.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_40.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_50.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_60.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_70.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_80.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_90.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_100.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_120.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_150.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_200.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_250.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_300.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_350.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_400.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_500.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_600.pkl',
    #     'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_749.pkl',
    #     ]
    
    env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'dynamic_1.0',
    'routes_group': 'multi_bruno_3_full'
    }

    # env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
    #                     terminal_configs=terminal_configs, host="localhost", port=2001,
    #                     seed=2021, no_rendering=False, **env_configs)
    # env = RlBirdviewWrapper(env)






    models = [
        # 'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_1.pkl',
        # 'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_20.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_30.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_40.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_50.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_60.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_70.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_80.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_90.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_100.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_120.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_150.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_200.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_250.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_300.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_350.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_400.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_500.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_600.pkl',
        'model_pytorch_multi_behavior_cloning/gail_experts_nroutes1_neps1_ce06_ep_749.pkl',
        ]

    env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                        terminal_configs=terminal_configs, host="localhost", port=2020,
                        seed=2021, no_rendering=False, **env_configs, spawn_point=spawn_point)
    env = RlBirdviewWrapper(env)
    
    model = Model_cnn_BC(x_shape=(192, 192, 4), n_hidden=128, cnn_out_dim=2).to(device)

    # env_configs = {
    #     'carla_map': 'Town01',
    #     'num_zombie_vehicles': [0, 150],
    #     'num_zombie_walkers': [0, 300],
    #     'weather_group': 'dynamic_1.0'
    #     }

    # env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
    #                 terminal_configs=terminal_configs, host='localhost', port=2001,
    #                 seed=np.random.randint(1, 3001), 
    #                 no_rendering=True, **env_configs, spawn_point=spawn_point)
    # env = RlBirdviewWrapper(env)
    # env = SubprocVecEnv([env])
    # env = env_maker_multimodality
    for model_path in models:
        model.load_state_dict(torch.load(model_path))
        for i in range(12, 100):
            # eval_video_path = diff_bc_video+f'/diff_bc_eval_749_{i}.mp4'
            eval_video_path = diff_bc_video + model_path.split('/')[-1].split('.')[0] + f'_{i}' + '.mp4'
            evaluate_policy(
                env=env,
                model=model.to(device),
                video_path=eval_video_path,
                device=device,
                observation_type=observation_type,
                max_eval_steps=300,
                architecture='mse')
            # object = FrontCameraMovieMaker(path=route_path, name_index=str(i)+f'_ep_0{j}')
            # object.save_record()