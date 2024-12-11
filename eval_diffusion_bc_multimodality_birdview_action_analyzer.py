import numpy as np
import torch
import time
import os
import math
import pandas as pd
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_birdview_wrapper import RlBirdviewWrapper
# from carla_gym.envs import EndlessEnv, EndlessFixedSpawnEnv, LeaderboardEnv
from carla_gym.envs import EndlessFixedSpawnEnv
from models import Model_cnn_mlp, Model_Cond_Diffusion, Model_cnn_mlp_resnet, Model_cnn_mlp_original
from data_collect import reward_configs, terminal_configs, obs_configs
from data_preprocessing import DataHandler, FrontCameraMovieMakerArray
from models_bc import Model_cnn_BC
import matplotlib.pyplot as plt


env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'dynamic_1.0',
    'routes_group': 'multi_bruno_3_full'
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
    'x':94.6903991699219,
    'y':194.78451538085938,
    'yaw':179.83230590820312,
    'z':0.0
}


def handle_obs(obs, observation_type, embedding):
    obs = DataHandler().preprocess_images(obs, observation_type=observation_type , eval=True, embedding=embedding)
    return obs

def eval_policy_multimodality(env, model, img_path, device, max_eval_steps=1, observation_type='birdview', architecture='diffusion', extra_steps=0, embedding=Model_cnn_mlp):
    model = model.eval()
    n_step = 0
    actions_list = []
    while n_step < max_eval_steps:
        obs = env.reset()
        obs = handle_obs(obs, observation_type, embedding=embedding)
        if architecture == 'diffusion':
            actions = model.sample_extra(torch.tensor(obs).float().to(device), extra_steps=extra_steps).to(device)[0]
        elif architecture == 'mse':
            actions = model(torch.tensor(obs).float().to(device)).to(device)[0]
        n_step += 1
        actions_list.append(list(actions.detach().numpy()))
        print(n_step)
    create_and_save_histogram(actions_list, img_path)
    return actions_list


def create_and_save_histogram(actions_list, img_path):

    data = np.array(actions_list)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(data[:, 0], bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title("Acceleration Histogram")
    axes[0].set_xlabel("Values")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0,1)

    axes[1].hist(data[:, 1], bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_title("Steering Histogram")
    axes[1].set_xlabel("Values")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(-1,1)

    plt.tight_layout()
    plt.savefig(img_path)
    print("Histograma salvo como 'histogramas_BC.png'")





# from PIL import Image
# Image.fromarray(np.transpose(obs['birdview'].astype(dtype=np.uint8), (1,2,0))).save('imagem.png')

if __name__ == '__main__':
    diff_bc_video = 'diff_bc_video_(not_diffuser)/multi_birdview/'
    diff_bc_video = 'diff_bc_video_(not_diffuser)/birdview/teste_3/'
    diff_bc_video = 'diff_bc_video_(diffuser)/front/resnet18/teste/'
    diff_bc_video = 'diff_bc_video_(diffuser)/birdview/Diffusion_BC_Multi_Simple_03_067e/'

    os.makedirs(diff_bc_video, exist_ok=True)

    device = 'cpu'
    net_type = 'transformer'
    observation_type = 'birdview'

    x_shape = (192, 192, 4)
    y_dim = 2
    embed_dim = 128
    n_hidden = 128


    nn_model = Model_cnn_mlp_original(
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
        x_dim=x_shape,
        y_dim=2,
        drop_prob=0.0,
        guide_w=0.0,)

    env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                        terminal_configs=terminal_configs, host="localhost", port=2000,
                        seed=2021, no_rendering=False, **env_configs, spawn_point=spawn_point_action_histogram)
    env = RlBirdviewWrapper(env)

    models = [
        # 'model_pytorch/Diffusion_BC_Multi_Simple_01/Model_cnn_BC_gail_experts_multi_bruno_3_simples_birdviewt_BC_067e_ep_80.pkl',
        'model_pytorch/Diffusion_BC_Multi_Simple_02/Model_cnn_BC_gail_experts_multi_bruno_3_simples_birdviewt_BC_067e_ep_80.pkl'
    ]





    # models = [
    #     'model_pytorch/BC_Multi_Simple_01/Model_cnn_BC_gail_experts_multi_bruno_3_simples_birdviewt_BC_067e_ep_20.pkl',
    # ]
    # model = Model_cnn_BC(x_shape=(192, 192, 4), n_hidden=128, cnn_out_dim=2).to(device)
    # -----------------------------------------------------------------------------------------
    extra_steps_list = [0,8]
    for extra_steps in extra_steps_list:
        for model_path in models:
            model.load_state_dict(torch.load(model_path))
            for i in range(100):
                diff_bc_video = f'test_2/diff_bc_video_(diffuser)/birdview/new_arch/town01_multimodality_t_intersection_simples_extra_steps/{model_path.split("/")[1]}/town01_multimodality_t_intersection_simples_{extra_steps}_extra_steps/'
                diff_bc_video_2 = diff_bc_video + model_path.split('/')[-2] + '/'
                os.makedirs(diff_bc_video_2, exist_ok=True)
                eval_video_path = diff_bc_video_2 + model_path.split('/')[-1].split('.')[0] + f'_{i}' + '.mp4'
                eval_policy_multimodality(
                    env=env,
                    model=model.to(device),
                    img_path='',
                    device=device,
                    observation_type=observation_type,
                    max_eval_steps=1000,
                    architecture='diffusion',
                    extra_steps=extra_steps,
                    embedding='Model_cnn_mlp')


















# def evaluate_policy(env, model, video_path, device, max_eval_steps=3000, observation_type='birdview',  architecture='diffusion', movie=True, extra_steps=0, embedding=Model_cnn_mlp):
#     model = model.eval()
#     t0 = time.time()
#     obs = env.reset()
#     previous_position = obs['gnss']
#     obs = handle_obs(obs, observation_type, embedding=embedding)
#     n_step = 0
#     env_done = False
#     list_render = []
#     ep_stat_buffer = []
#     route_completion_buffer = pd.DataFrame(columns=['step', 'simulation_time', 'route_completed_in_m', 'route_length_in_m', 'is_route_completed'])
#     route_infraction = pd.DataFrame(columns=['c_blocked', 'c_lat_dist', 'c_collision', 'collision', 'c_collision_px', 'timeout', 'info_dict', 'lat_dist', 'thresh_lat_dist'])
#     route_infraction_2 = pd.DataFrame(columns=['collisions_layout','collisions_vehicle','collisions_pedestrian','collisions_others','route_deviation','wrong_lane','outside_lane','run_red_light','encounter_stop','stop_infraction'])
#     list_render_front = []
#     list_render_left = []
#     list_render_right = []
#     list_render_birdview = []
#     list_gnss = []
#     ep_dict = {}
#     ep_dict['actions'] = []
#     ep_dict['state'] = []
#     distance_traveled = 0

#     while n_step < max_eval_steps:
#         if architecture == 'diffusion':
#             actions = model.sample_extra(torch.tensor(obs).float().to(device), extra_steps=extra_steps).to(device)[0]
#         elif architecture == 'mse':
#             actions = model(torch.tensor(obs).float().to(device)).to(device)[0]
#         obs_clean, reward, done, info = env.step(np.array(actions.detach().cpu()))

#         distance_traveled += calculate_distance_traveled(obs_clean['gnss'], previous_position)
#         # previous_position = obs_clean['gnss']

#         new_row = pd.DataFrame([info['route_completion']])
#         route_completion_buffer = pd.concat([route_completion_buffer, new_row], ignore_index=True)

#         filtered_data = {key: info['terminal_debug'][key] for key in route_infraction.columns if key in info['terminal_debug']}
#         filtered_data_2 = {key: info[key] for key in route_infraction_2.columns if key in info}
#         new_row = pd.DataFrame([filtered_data])
#         new_row_2 = pd.DataFrame([filtered_data_2])
#         route_infraction = pd.concat([route_infraction, new_row], ignore_index=True)
#         route_infraction_2 = pd.concat([route_infraction_2, new_row_2], ignore_index=True)

#         route_infraction_total = pd.concat([route_infraction, route_infraction_2], axis=1)

#         obs = handle_obs(obs_clean, observation_type, embedding)
        
#         if True:
#             list_render.append(np.transpose(obs_clean['central_rgb'], (1,2,0)))
#             list_render_front.append(np.transpose(obs_clean['central_rgb'], (1,2,0)))
#             list_render_right.append(np.transpose(obs_clean['right_rgb'], (1,2,0)))
#             list_render_left.append(np.transpose(obs_clean['left_rgb'], (1,2,0)))
#             list_render_birdview.append(np.transpose(obs_clean['birdview'], (1,2,0)))
#             list_gnss.append(info['gnss'])
#             ep_dict['state'].append(np.transpose(obs_clean['birdview'], (1,2,0)))
#             ep_dict['actions'].append([actions[0].item(), actions[1].item()])
#         else:
#             list_render.append(env.render(mode='rgb_array'))
#         n_step += 1
#         env_done = done
        
#         print(f'n_step: {n_step} ---- distance_traveled: {distance_traveled}')

#         for i in np.where(done)[0]:
#             break

#     if True:
#         if movie:
#             movie_maker = FrontCameraMovieMakerArray(video_path=video_path,
#                                                     front_array=list_render_front,
#                                                     left_array=list_render_left,
#                                                     right_array=list_render_right,
#                                                     birdview_array=list_render_birdview)
#             movie_maker.save_record(text=f'{distance_traveled}')

#         gnss_path = video_path[:-3]+'txt'
#         route_completion_data_path = video_path[:-3]+'csv'
#         actions_observation_path = video_path[:-3]+'json'

#         route_info = pd.concat([route_completion_buffer, route_infraction_total], axis=1)
#         route_info.to_csv(route_completion_data_path, index=False)

#         np.savetxt(gnss_path, list_gnss)
#         ep_df = pd.DataFrame(ep_dict)
#         ep_df.to_json(actions_observation_path)
#     return distance_traveled

# def update_dataframe(df, route_completion):
#     df = df.append(route_completion, ignore_index=True)

# def env_maker(multimodality):
#     if multimodality:
#         return env_maker_multimodality
#     else:
#         return env_maker_fixed_route

# def env_maker_fixed_route():

#     env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
#                     terminal_configs=terminal_configs, host='localhost', port=2001,
#                     seed=100, 
#                     no_rendering=True, **env_configs)
#     env = RlBirdviewWrapper(env)
#     return env

# def env_maker_multimodality():

#     # env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
#     #                 terminal_configs=terminal_configs, host='localhost', port=3001,
#     #                 seed=np.random.randint(1, 3001), 
#     #                 no_rendering=True, **env_configs)

#     env_configs = {
#         'carla_map': 'Town01',
#         'num_zombie_vehicles': [0, 150],
#         'num_zombie_walkers': [0, 300],
#         'weather_group': 'dynamic_1.0'
#         }

#     # env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
#     #                 terminal_configs=terminal_configs, host='localhost', port=2001,
#     #                 seed=np.random.randint(1, 3001), 
#     #                 no_rendering=True, **env_configs, spawn_point=spawn_point)

#     env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
#                     terminal_configs=terminal_configs, host='localhost', port=2001,
#                     seed=np.random.randint(1, 3001), 
#                     no_rendering=True, **env_configs)
#     env = RlBirdviewWrapper(env)
#     return env


# def calculate_distance_traveled(current_position, previous_position):
#     """
#     Calcula a distância percorrida por um veículo desde a última posição registrada.
    
#     Args:
#         vehicle: Objeto do veículo no CARLA.
#         previous_position: Última posição registrada do veículo (carla.Location).
    
#     Returns:
#         distance: Distância percorrida desde a última posição (float).
#         current_position: Posição atual do veículo (carla.Location).
#     """
    
#     if previous_position is None:
#         # Primeira chamada: sem deslocamento
#         return 0.0, current_position

#     # Calcular o deslocamento 3D
#     dx = current_position[0] - previous_position[0]
#     dy = current_position[1] - previous_position[1]
#     dz = current_position[2] - previous_position[2]
#     distance = math.sqrt(dx**2 + dy**2)
    
#     return distance


