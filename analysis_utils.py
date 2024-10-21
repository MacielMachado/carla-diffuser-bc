import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def process_files(path_to_data_folder):
    data_list = []

    # Percorrer todas as subpastas
    for root, dirs, files in tqdm(os.walk(path_to_data_folder)):
        for file_name in files:
            if file_name.endswith('.csv'):
                # Extrair M do nome da subpasta
                folder_name = os.path.basename(root)
                
                if folder_name.startswith('BC_Multi_Simple_'):
                    BC = 0
                elif folder_name.startswith('Diffusion_BC_Multi_Simple_'):
                    BC = 1
                else:
                    continue
                
                # Extrair M (número após o último '_')
                M = int(folder_name.split('_')[-1])

                # Extrair Ech e Ep do nome do arquivo .csv
                # Padrão do arquivo: 'gail_experts_nroutes1_neps1_067e_ep_ECH_EP.csv'
                parts = file_name.split('_')
                Ech = int(parts[-2])  # Valor entre 'ep_' e o último '_'
                Ep = int(parts[-1].replace('.csv', ''))  # Valor após o último '_'
                
                # Achar o arquivo .txt correspondente (mesmo Ech e Ep)
                txt_file_name = f"Model_cnn_BC_gail_experts_multi_bruno_3_simples_birdviewt_BC_067e_ep_{Ech}_{Ep}.txt"
                txt_file_path = os.path.join(root, txt_file_name)
                
                if not os.path.exists(txt_file_path):
                    print(f"Arquivo .txt não encontrado: {txt_file_path}")
                    continue

                # Ler x e y do arquivo .txt (primeira e segunda colunas)
                with open(txt_file_path, 'r') as f:
                    lines = f.readlines()
                    x = [float(line.split()[0]) for line in lines]  # Primeira coluna
                    y = [float(line.split()[1]) for line in lines]  # Segunda coluna

                # Ler os dados do arquivo .csv
                csv_file_path = os.path.join(root, file_name)
                csv_data = pd.read_csv(csv_file_path)

                # Verificar se todas as colunas necessárias estão no .csv
                required_columns = ['simulation_time', 'route_completed_in_m', 'route_length_in_m', 'is_route_completed', 
                                    'c_blocked', 'c_lat_dist', 'c_collision', 'collision', 'c_collision_px', 'timeout', 
                                    'info_dict', 'lat_dist', 'thresh_lat_dist', 'collisions_layout','collisions_vehicle',
                                    'collisions_pedestrian','collisions_others','route_deviation','wrong_lane','outside_lane',
                                    'run_red_light','encounter_stop','stop_infraction']

                if not all(col in csv_data.columns for col in required_columns):
                    print(f"Arquivo .csv não contém todas as colunas necessárias: {csv_file_path}")
                    continue

                # Extrair os valores das colunas do .csv
                simulation_time = csv_data['simulation_time'].values
                route_completed_in_m = csv_data['route_completed_in_m'].values
                route_length_in_m = csv_data['route_length_in_m'].values
                is_route_completed = csv_data['is_route_completed'].values
                c_blocked = csv_data['c_blocked'].values
                c_lat_dist = csv_data['c_lat_dist'].values
                c_collision = csv_data['c_collision'].values
                collision = csv_data['collision'].values
                c_collision_px = csv_data['c_collision_px'].values
                timeout = csv_data['timeout'].values
                info_dict = csv_data['info_dict'].values
                lat_dist = csv_data['lat_dist'].values
                thresh_lat_dist = csv_data['thresh_lat_dist'].values

                collisions_layout = csv_data['collisions_layout'].values
                collisions_vehicle = csv_data['collisions_vehicle'].values
                collisions_pedestrian = csv_data['collisions_pedestrian'].values
                collisions_others = csv_data['collisions_others'].values
                route_deviation = csv_data['route_deviation'].values
                wrong_lane = csv_data['wrong_lane'].values
                outside_lane = csv_data['outside_lane'].values
                run_red_light = csv_data['run_red_light'].values
                encounter_stop = csv_data['encounter_stop'].values
                stop_infraction = csv_data['stop_infraction'].values

                # Adicionar ao array
                data_list.append([BC, M, Ech, Ep, [x, y, simulation_time, route_completed_in_m, 
                                                   route_length_in_m, is_route_completed, c_blocked, 
                                                   c_lat_dist, c_collision, collision, c_collision_px, 
                                                   timeout, info_dict, lat_dist, thresh_lat_dist,
                                                   collisions_layout, collisions_vehicle, collisions_pedestrian,
                                                   collisions_others, route_deviation, wrong_lane, outside_lane,
                                                   run_red_light, encounter_stop, stop_infraction]])

    # Converter a lista para um numpy array
    data_array = np.array(data_list, dtype=object)
    return data_array

def process_files(path_to_data_folder):
    data_list = []

    # Percorrer todas as subpastas
    for root, dirs, files in tqdm(os.walk(path_to_data_folder)):
        for file_name in files:
            if file_name.endswith('.csv'):
                # Extrair M do nome da subpasta
                folder_name = os.path.basename(root)
                
                if folder_name.startswith('BC'):
                    BC = 0
                elif folder_name.startswith('Diffusion_BC'):
                    BC = 1
                else:
                    continue
                
                # Extrair M (número após o último '_')
                M = int(folder_name.split('_')[-1])

                # Extrair Ech e Ep do nome do arquivo .csv
                # Padrão do arquivo: 'gail_experts_nroutes1_neps1_067e_ep_ECH_EP.csv'
                parts = file_name.split('_')
                Ech = int(parts[-2])  # Valor entre 'ep_' e o último '_'
                Ep = int(parts[-1].replace('.csv', ''))  # Valor após o último '_'
                
                # Achar o arquivo .txt correspondente (mesmo Ech e Ep)
                # txt_file_name_1 = f"town01_fixed_route_without_trajectory_birdview_5bbd_ep_{Ech}_{Ep}.txt"
                txt_file_name_1 = f"town01_fixed_route_without_trajectory_birdview_5bbd_ep_{Ech}_{Ep}.txt"
                txt_file_path_1 = os.path.join(root, txt_file_name_1)

                # txt_file_name_2 = f"town01_fixed_route_without_trajectory_birdview_11a6_ep_{Ech}_{Ep}.txt"
                txt_file_name_2 = f"town01_Diff_BC_multi_without_trajectory_birdview_GKC_speed_e1d8_ep_{Ech}_{Ep}.txt"
                txt_file_path_2 = os.path.join(root, txt_file_name_2)
                
                # if not os.path.exists(txt_file_path_1) or not os.path.exists(txt_file_path_2):

                # Ler x e y do arquivo .txt (primeira e segunda colunas)
                if os.path.exists(txt_file_path_1):
                    with open(txt_file_path_1, 'r') as f:
                        lines = f.readlines()
                        x = [float(line.split()[0]) for line in lines]  # Primeira coluna
                        y = [float(line.split()[1]) for line in lines]  # Segunda coluna
                elif os.path.exists(txt_file_path_2):
                    with open(txt_file_path_2, 'r') as f:
                        lines = f.readlines()
                        x = [float(line.split()[0]) for line in lines]  # Primeira coluna
                        y = [float(line.split()[1]) for line in lines]  # Segunda coluna
                else:
                    print(f"Arquivo .txt não encontrado: {txt_file_path_1} ou {txt_file_path_2}")
                    continue


                # Ler os dados do arquivo .csv
                csv_file_path = os.path.join(root, file_name)
                csv_data = pd.read_csv(csv_file_path)

                # Verificar se todas as colunas necessárias estão no .csv
                required_columns = ['simulation_time', 'route_completed_in_m', 'route_length_in_m', 'is_route_completed', 
                                    'c_blocked', 'c_lat_dist', 'c_collision', 'collision', 'c_collision_px', 'timeout', 
                                    'info_dict', 'lat_dist', 'thresh_lat_dist', 'collisions_layout','collisions_vehicle',
                                    'collisions_pedestrian','collisions_others','route_deviation','wrong_lane','outside_lane',
                                    'run_red_light','encounter_stop','stop_infraction']

                if not all(col in csv_data.columns for col in required_columns):
                    print(f"Arquivo .csv não contém todas as colunas necessárias: {csv_file_path}")
                    continue

                # Extrair os valores das colunas do .csv
                simulation_time = csv_data['simulation_time'].values
                route_completed_in_m = csv_data['route_completed_in_m'].values
                route_length_in_m = csv_data['route_length_in_m'].values
                is_route_completed = csv_data['is_route_completed'].values
                c_blocked = csv_data['c_blocked'].values
                c_lat_dist = csv_data['c_lat_dist'].values
                c_collision = csv_data['c_collision'].values
                collision = csv_data['collision'].values
                c_collision_px = csv_data['c_collision_px'].values
                timeout = csv_data['timeout'].values
                info_dict = csv_data['info_dict'].values
                lat_dist = csv_data['lat_dist'].values
                thresh_lat_dist = csv_data['thresh_lat_dist'].values

                collisions_layout = csv_data['collisions_layout'].values
                collisions_vehicle = csv_data['collisions_vehicle'].values
                collisions_pedestrian = csv_data['collisions_pedestrian'].values
                collisions_others = csv_data['collisions_others'].values
                route_deviation = csv_data['route_deviation'].values
                wrong_lane = csv_data['wrong_lane'].values
                outside_lane = csv_data['outside_lane'].values
                run_red_light = csv_data['run_red_light'].values
                encounter_stop = csv_data['encounter_stop'].values
                stop_infraction = csv_data['stop_infraction'].values

                # Adicionar ao array
                data_list.append([BC, M, Ech, Ep, [x, y, simulation_time, route_completed_in_m, 
                                                   route_length_in_m, is_route_completed, c_blocked, 
                                                   c_lat_dist, c_collision, collision, c_collision_px, 
                                                   timeout, info_dict, lat_dist, thresh_lat_dist,
                                                   collisions_layout, collisions_vehicle, collisions_pedestrian,
                                                   collisions_others, route_deviation, wrong_lane, outside_lane,
                                                   run_red_light, encounter_stop, stop_infraction]])

    # Converter a lista para um numpy array
    data_array = np.array(data_list, dtype=object)
    return data_array

# Função para contar as infrações e plotar os histogramas
def plot_histograms(data):
    # Dicionário para armazenar contagens de infrações por BC, M e Ech
    infraction_counts = defaultdict(lambda: defaultdict(int))
    
    for entry in data:
        BC, M, Ech, Ep, values = entry
        # Valores de infração
        is_route_completed = values[5]
        c_blocked = values[6]
        c_lat_dist = values[7]
        c_collision = values[8]
        c_collision_px = values[10]
        timeout = values[11]

        # Verificar se é uma execução sem infração (todas falsas)
        no_infraction = (sum(is_route_completed) == 0 and sum(c_blocked) == 0 
                         and sum(c_lat_dist) == 0 and sum(c_collision) == 0
                         and sum(c_collision_px) == 0 and sum(timeout) == 0)
        
        if no_infraction:
            infraction_counts[(BC, M, Ech)]['No Infraction'] += 1
        else:
            # Contar cada tipo de infração
            if sum(is_route_completed):
                infraction_counts[(BC, M, Ech)]['is_route_completed'] += 1
            if sum(c_lat_dist) and sum(c_collision):
                infraction_counts[(BC, M, Ech)]['c_lat_dist_and_c_collision'] += 1
            if sum(c_blocked):
                infraction_counts[(BC, M, Ech)]['c_blocked'] += 1
            if sum(c_lat_dist):
                infraction_counts[(BC, M, Ech)]['c_lat_dist'] += 1
            if sum(c_collision):
                infraction_counts[(BC, M, Ech)]['c_collision'] += 1
            if sum(c_collision_px):
                infraction_counts[(BC, M, Ech)]['c_collision_px'] += 1
            if sum(timeout):
                infraction_counts[(BC, M, Ech)]['timeout'] += 1

    # Gerar os histogramas para cada combinação de BC, M e Ech
    for (BC, M, Ech), infractions in infraction_counts.items():
        # Nome das infrações
        infraction_types = ['No Infraction', 'is_route_completed', 'c_blocked',
                            'c_lat_dist', 'c_collision', 'c_collision_px',
                            'timeout']
        
        # Obter as contagens das infrações
        counts = [infractions[infraction] for infraction in infraction_types]

        # Plotar o histograma
        plt.figure(figsize=(10, 6))
        plt.bar(infraction_types, counts, color='b')
        plt.title(f'Infractions for BC={BC}, M={M}, Ech={Ech}')
        plt.xlabel('Infraction Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Função para contar as infrações e plotar os histogramas por BC + Ech
def plot_histograms_by_bc_ech(data):
    # Dicionário para armazenar contagens de infrações por BC e Ech (agregando M)
    infraction_counts = defaultdict(lambda: defaultdict(int))
    
    for entry in data:
        BC, M, Ech, Ep, values = entry
        print(len(values))
        # Valores de infração
        is_route_completed = values[4]
        c_blocked = values[5]
        c_lat_dist = values[6]
        c_collision = values[7]
        collision = values[8]
        c_collision_px = values[9]
        timeout = values[10]
        collisions_layout = values[14]
        collisions_vehicle = values[15]
        collisions_pedestrian = values[16]
        collisions_others = values[17]
        route_deviation = values[18]
        wrong_lane = values[19]
        outside_lane = values[20]
        run_red_light = values[21]
        encounter_stop = values[22]
        stop_infraction = values[23]

        # Verificar se é uma execução sem infração (todas falsas)
        c_blocked_sum = sum(c_blocked)
        c_lat_dist_sum = sum(c_lat_dist)
        c_collision_sum = sum(c_collision)
        collision_sum = sum(collision)
        c_collision_px_sum = sum(c_collision_px)
        timeout_sum = sum(timeout)
        collisions_layout_sum = sum(collisions_layout)
        collisions_vehicle_sum = sum(collisions_vehicle)
        collisions_pedestrian_sum = sum(collisions_pedestrian)
        collisions_others_sum = sum(collisions_others)
        route_deviation_sum = sum(route_deviation)
        wrong_lane_sum = sum(wrong_lane)
        outside_lane_sum = sum(outside_lane)
        run_red_light_sum = sum(run_red_light)
        encounter_stop_sum = sum(encounter_stop)
        stop_infraction_sum = sum(stop_infraction)
        # print(c_blocked)
        no_infraction = (timeout_sum+collision_sum+c_collision_sum+c_lat_dist_sum+c_blocked_sum+collisions_layout_sum+collisions_vehicle_sum+collisions_pedestrian_sum+collisions_others_sum+route_deviation_sum+wrong_lane_sum+outside_lane_sum+run_red_light_sum+encounter_stop_sum+stop_infraction_sum) == 0
        
        # Agrupar por BC e Ech
        if no_infraction:
            infraction_counts[(BC, Ech)]['No Infraction'] += 1
        else:
            # Contar cada tipo de infração
            if c_blocked_sum:
                infraction_counts[(BC, Ech)]['c_blocked'] += 1
            elif c_lat_dist_sum:
                infraction_counts[(BC, Ech)]['c_lat_dist'] += 1
            elif c_collision_sum:
                infraction_counts[(BC, Ech)]['c_collision'] += 1
            elif collision_sum:
                infraction_counts[(BC, Ech)]['collision'] += 1
            elif timeout_sum:
                infraction_counts[(BC, Ech)]['timeout'] += 1
            elif collisions_layout_sum:
                infraction_counts[(BC, Ech)]['collisions_layout'] += 1
            elif collisions_vehicle_sum:
                infraction_counts[(BC, Ech)]['collisions_vehicle'] += 1
            elif collisions_pedestrian_sum:
                infraction_counts[(BC, Ech)]['collisions_pedestrian'] += 1
            elif collisions_others_sum:
                infraction_counts[(BC, Ech)]['collisions_others'] += 1
            elif route_deviation_sum:
                infraction_counts[(BC, Ech)]['route_deviation'] += 1
            elif wrong_lane_sum:
                infraction_counts[(BC, Ech)]['wrong_lane'] += 1
            elif outside_lane_sum:
                infraction_counts[(BC, Ech)]['outside_lane'] += 1
            elif run_red_light_sum:
                infraction_counts[(BC, Ech)]['run_red_light'] += 1
            elif encounter_stop_sum:
                infraction_counts[(BC, Ech)]['encounter_stop'] += 1
            elif stop_infraction_sum:
                infraction_counts[(BC, Ech)]['stop_infraction'] += 1

    # Ordenar os histogramas por BC e Ech
    sorted_keys = sorted(infraction_counts.keys())

    # Gerar os histogramas para cada combinação de BC e Ech
    for (BC, Ech) in sorted_keys:
        infractions = infraction_counts[(BC, Ech)]
        # Nome das infrações
        infraction_types = ['No Infraction', 'c_blocked', 'c_lat_dist',
                            'c_collision', 'collision', 'timeout',
                            'collisions_layout', 'collisions_vehicle',
                            'collisions_pedestrian', 'collisions_others',
                            'route_deviation', 'wrong_lane', 'outside_lane',
                            'run_red_light', 'encounter_stop', 'stop_infraction']
        
        # Obter as contagens das infrações
        counts = [infractions[infraction] for infraction in infraction_types]

        # Plotar o histograma
        plt.figure(figsize=(10, 6))
        plt.bar(infraction_types, counts, color='g')
        plt.title(f'Infractions for BC={BC}, Ech={Ech} (Aggregated M)')
        plt.xlabel('Infraction Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.ylim([0,50])
        plt.tight_layout()
        plt.grid()
        plt.show()





if __name__ == '__main__':
    # path_to_data_folder = "diff_bc_video_(diffuser)/birdview/town01_multimodality_t_intersection_simples"
    # data = np.load(path_to_data_folder + '/data.npy', allow_pickle=True)
    # plot_histograms(data)

    # path_to_data_folder = "diff_bc_video_(diffuser)/birdview/town01_multimodality_t_intersection_simples_2"
    # data = np.load(path_to_data_folder + '/data.npy', allow_pickle=True)
    # plot_histograms_by_bc_ech(data)

    # path_to_data_folder = "diff_bc_video_(diffuser)/birdview/town01_multimodality_t_intersection_simples_with_actions_with_fixed_route"
    # path_to_data_folder = "diff_bc_video_(diffuser)/birdview/town01_multimodality_t_intersection_simples_GKC"
    path_to_data_folder = 'diff_bc_video_(diffuser)/birdview/town01_multimodality_t_intersection_simples_with_actions_with_fixed_route_32_extra_steps'
    output_file = path_to_data_folder + '/data.npy'
    result = process_files(path_to_data_folder)
    np.save(output_file, result)
    print(f"Array salvo no arquivo {output_file}")




    # path_to_data_folder = "diff_bc_video_(diffuser)/birdview/town01_multimodality_t_intersection_simples_2"
    # output_file = path_to_data_folder + '/data_3.npy'
    # result = process_files(path_to_data_folder)
    # np.save(output_file, result)
    # print(f"Array salvo no arquivo {output_file}")

    # print(result)
