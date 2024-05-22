import numpy as np
import torch
import time
import os
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from carla_gym.envs import EndlessEnv, EndlessFixedSpawnEnv
from models import Model_cnn_mlp, Model_Cond_Diffusion
from data_collect import reward_configs, terminal_configs, obs_configs
from data_preprocessing import DataHandler, FrontCameraMovieMaker

env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}

def handle_obs(obs):
    obs = DataHandler().preprocess_images(obs, observation_type='birdview', eval=True)
    return obs

def evaluate_policy(env, model, video_path, device, min_eval_steps=3001, origin='birdview'):
    model = model.eval()
    t0 = time.time()
    # for i in range(env.num_envs):
    #     env.set_attr('eval_mode', True, indices=1)
    obs = env.reset()
    obs = handle_obs(obs)
    n_step = 0
    env_done = np.array([False]*env.num_envs)
    list_render = []
    ep_stat_buffer = []
    route_completion_buffer = []

    while n_step < min_eval_steps and np.sum(env_done) == 0:
        actions = model.sample(torch.tensor(obs).float().to(device)).to(device)
        obs, reward, done, info = env.step(np.array(actions.detach().cpu()))
        obs = handle_obs(obs)
    
        list_render.append(env.render(mode='rgb_array'))
        n_step += 1
        env_done |= done
        
        print(f'n_step: {n_step}')

    # conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
    encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
    for im in list_render:
        encoder.capture_frame(im)
    encoder.close()

def env_maker():

    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=3001,
                    seed=np.random.randint(1, 3001), 
                    no_rendering=True, **env_configs)
    env = RlBirdviewWrapper(env)
    return env


if __name__ == '__main__':
    diff_bc_video = 'diff_bc_video_fixed_full_100/'
    os.makedirs(diff_bc_video, exist_ok=True)

    device = 'cpu'
    net_type = 'transformer'
    x_shape = (192, 192, 4)
    y_dim = 2
    embed_dim = 128
    n_hidden = 128

    nn_model = Model_cnn_mlp(
        x_shape,
        n_hidden,
        y_dim,
        embed_dim=embed_dim,
        net_type=net_type,
        cnn_out_dim=4608).to(device)

    model = Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=20,
        device=device,
        x_dim=(192, 192, 4),
        y_dim=2,
        drop_prob=0.0,
        guide_w=0.0,)
    
    # model_path = 'model_pytorch/gail_experts_nroutes1_neps1_12bf_ep_749.pkl'
    models = [
            'model_pytorch/fixed/gail_experts_nroutes1_neps1_b842_ep_500.pkl',
            'model_pytorch/fixed/gail_experts_nroutes1_neps1_63f5_ep_250.pkl',
            'model_pytorch/fixed/gail_experts_nroutes1_neps1_b668_ep_150.pkl',
            'model_pytorch/fixed/gail_experts_nroutes1_neps1_b668_ep_80.pkl',
            'model_pytorch/fixed/gail_experts_nroutes1_neps1_a51b_ep_40.pkl',
            'model_pytorch/fixed/gail_experts_nroutes1_neps1_a51b_ep_20.pkl',
            'model_pytorch/fixed/gail_experts_nroutes1_neps1_a51b_ep_1.pkl',
            ]

    # models = ['model_pytorch_multi_full_front/gail_experts_nroutes1_neps1_8d42_ep_20.pkl',]

    env = SubprocVecEnv([env_maker])

    for model_path in models:
        model.load_state_dict(torch.load(model_path, map_location=device))
        for i in range(100):
            # eval_video_path = diff_bc_video+f'/diff_bc_eval_749_{i}.mp4'
            eval_video_path = diff_bc_video + model_path.split('/')[-1].split('.')[0] + f'_{i}' + '.mp4'
            evaluate_policy(
                env=env,
                model=model,
                video_path=eval_video_path,
                device=device,
                origin='birdview')
            # object = FrontCameraMovieMaker(path=route_path, name_index=str(i)+f'_ep_0{j}')
            # object.save_record()