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
# from carla_gym.envs import EndlessEnv, EndlessFixedSpawnEnv, LeaderboardEnv
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

def gerar_histogramas(actions, save_path):
    coluna_a = [sublista[0] for sublista in actions]
    coluna_b = [sublista[1] for sublista in actions]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(coluna_a, bins=10, color='blue', alpha=0.7)
    axs[0].set_title('Acceleration')

    axs[0].set_xlim([0, 1])
    # axs[0].set_ylim([0, 500])

    axs[1].hist(coluna_b, bins=10, color='green', alpha=0.7)
    axs[1].set_title('Steering')

    axs[1].set_xlim([-1, 1])
    # axs[1].set_ylim([0, 500])

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    
    return fig

def evaluate_policy(env, model, video_path, device, max_eval_steps=3000, observation_type='birdview',  architecture='diffusion', movie=True, extra_steps=0, embedding=Model_cnn_mlp, persist_points = None, plotter=None, num_of_actions=100):
        
    model = model.eval()
    t0 = time.time()
    obs = env.reset()
    previous_position = obs['gnss']
    obs = handle_obs(obs, observation_type, embedding=embedding)
    n_step = 0
    ep_dict = {}
    ep_dict['actions'] = []
    ep_dict['state'] = []
    distance_traveled = 0
    actions_list = []
    while n_step < max_eval_steps:
        action_counter = 0
        while action_counter < num_of_actions:
            if architecture == 'diffusion':
                actions = model.sample_extra(torch.tensor(obs).float().to(device), extra_steps=extra_steps).to(device)[0]
            elif architecture == 'mse':
                actions = model(torch.tensor(obs).float().to(device)).to(device)[0]
            action_counter += 1
            actions_list.append(actions.cpu().detach().tolist())
            print(f'ep: {int(video_path[:-4].split("_")[-1])} ---- n_step: {n_step} ---- action_counter: {action_counter}')

        gerar_histogramas(actions=actions_list, save_path=video_path[:-6]+f"_step_{n_step}_histogram.png")

        obs_clean, reward, done, info = env.step(np.array(actions.detach().cpu()))

        obs = handle_obs(obs_clean, observation_type, embedding)

        n_step += 1        
        for i in np.where(done)[0]:
            break

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
    diff_bc_video = 'diff_bc_video_(not_diffuser)/multi_birdview/'
    diff_bc_video = 'diff_bc_video_(not_diffuser)/birdview/teste_3/'
    diff_bc_video = 'diff_bc_video_(diffuser)/front/resnet18/teste/'
    diff_bc_video = 'diff_bc_video_(diffuser)/birdview/Diffusion_BC_Multi_Simple_03_067e/'

    # diff_bc_video = 'diff_bc_video_(not_diffuser)/multi_birdview/'
    os.makedirs(diff_bc_video, exist_ok=True)

    device = 'cuda'
    net_type = 'transformer'
    observation_type = 'birdview'

    x_shape = (192, 192, 4)
    y_dim = 2
    embed_dim = 128
    n_hidden = 128

    env_configs = {
        'carla_map': 'Town01',
        'weather_group': 'dynamic_1.0',
        'routes_group': 'multi_bruno_3_full'
        }

    env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                        terminal_configs=terminal_configs, host="localhost", port=2020,
                        seed=2021, no_rendering=False, **env_configs, spawn_point=spawn_point_action_histogram)
    env = RlBirdviewWrapper(env)

    models_0 = ['model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_20.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_30.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_60.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_70.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_80.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_100.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_110.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_120.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_140.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_150.pkl',
              'model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_0/new_arch_bc03_ep_220.pkl',]
    
    models_2 = ['model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_2/new_arch_bc03_ep_80.pkl',]

    models_4 = ['model_pytorch/Diffusion_BC_Multi_Multiple_New_Arch/version_750_4/new_arch_bc03_ep_80.pkl',]
    
    models = models_0 + models_2 + models_4

    models = sort_by_filename_and_version(models)

    device = 'cuda'
    x_shape = (192, 192, 4)
    y_dim = 2
    embed_dim = 64
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
        x_dim=x_shape,
        y_dim=2,
        drop_prob=0.0,
        guide_w=0.0,)

    # -----------------------------------------------------------------------------------------
    extra_steps_list = [8]
    for extra_steps in extra_steps_list:
        for model_path in models:
            if int(model_path.split('.')[0].split('_')[-1]) % 20 != 0:
                continue
            model.load_state_dict(torch.load(model_path))
            persist_points = None
            diff_bc_video = f'diff_bc_video_(diffuser)/birdview/new_arch_carla_route_histogram/{model_path.split("/")[1]}_{extra_steps}_extra_steps/'
            diff_bc_video_2 = diff_bc_video + model_path.split('/')[-2] + '/'
            os.makedirs(diff_bc_video_2, exist_ok=True)
            eval_video_path = diff_bc_video_2 + model_path.split('/')[-1].split('.')[0] + f'_{0}' + '.mp4'
            if os.path.exists(eval_video_path[:-6]+"_map.png"):
                continue
            for i in range(25):
                diff_bc_video = f'diff_bc_video_(diffuser)/birdview/new_arch_carla_route_histogram/{model_path.split("/")[1]}_{extra_steps}_extra_steps/'
                diff_bc_video_2 = diff_bc_video + model_path.split('/')[-2] + '/'
                os.makedirs(diff_bc_video_2, exist_ok=True)
                eval_video_path = diff_bc_video_2 + model_path.split('/')[-1].split('.')[0] + f'_{i}' + '.mp4'
                evaluate_policy(
                                    env=env,
                                    model=model.to(device),
                                    video_path=eval_video_path,
                                    device=device,
                                    observation_type=observation_type,
                                    max_eval_steps=200,
                                    architecture='diffusion',
                                    movie=True,
                                    extra_steps=extra_steps,
                                    embedding='Model_cnn_mlp',
                                    persist_points = persist_points,)




