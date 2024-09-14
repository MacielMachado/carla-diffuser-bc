import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Função para plotar trajetórias (x, y) agregadas por BC e Ep
def plot_trajectories_by_bc_ep(data):
    # Dicionário para armazenar as trajetórias (x, y) por BC e Ep
    trajectories = defaultdict(lambda: defaultdict(list))

    # Percorrer os dados e agrupar as trajetórias por BC e Ep
    for entry in data:
        BC, M, Ech, Ep, values = entry
        x = values[0]  # x está na primeira coluna
        y = values[1]  # y está na segunda coluna
        
        # Variáveis para verificar infrações
        stop_infraction = values[24]
        encounter_stop = values[23]
        run_red_light = values[22]
        outside_lane = values[21]
        wrong_lane = values[20]
        collisions_layout = values[15]
        collisions_vehicle = values[16]
        collisions_pedestrian = values[17]
        c_blocked = values[6]
        c_lat_dist = values[7]
        c_collision = values[8]
        collision = values[9]
        c_collision_px = values[10]
        timeout = values[11]

        # Agregar trajetórias por BC e Ep
        trajectories[BC][Ech].append((x, y, M, Ep, stop_infraction, encounter_stop, run_red_light, outside_lane, wrong_lane, collisions_pedestrian, 
                                     collisions_layout, collisions_vehicle, c_blocked, c_lat_dist, c_collision, collision, c_collision_px, timeout))

    for BC in sorted(trajectories.keys()):  # Ordenar por BC
        for Ech in sorted(trajectories[BC].keys()):  # Ordenar por Ep
            plt.figure(figsize=(8, 6))
            
            # Contadores de trajetórias por cor
            red_count = 0
            blue_count = 0
            green_count = 0
            black_count = 0
            total_count = 0
            purple_marker_count = 0  # Contador para bolinhas roxas
            green_dict_list = []

            for (x, y, M, Ep, stop_infraction, encounter_stop, run_red_light, outside_lane, wrong_lane, collisions_pedestrian, collisions_layout, collisions_vehicle, 
                 c_blocked, c_lat_dist, c_collision, collision, c_collision_px, timeout) in trajectories[BC][Ech]:
                
                total_count += 1

                color = 'black'  # Padrão inicial

                purple_marker = None  # Para armazenar o índice onde ocorre infração específica

                # Verificar as infrações e definir cores conforme o comportamento anterior
                for i in range(len(x)):
                    if stop_infraction[i] == 1 or encounter_stop[i] == 1 or run_red_light[i] == 1 or collisions_pedestrian[i] == 1:
                        purple_marker = i  # Guardar o índice para desenhar a bolinha roxa

                    # Se ocorrer infrações fora do grupo, interromper a trajetória
                    if c_blocked[i] == 1 or c_lat_dist[i] == 1 or collisions_layout[i] == 1 or timeout[i]== 1 or outside_lane[i]==1 or wrong_lane[i]==1:
                        plot_until_index = i
                        # print("hi")
                        color_marker = 'red'
                        break

                if sum(c_blocked) + sum(c_lat_dist) + sum(collisions_layout) + sum(timeout) + sum(outside_lane) + sum(wrong_lane) == 0:
                    plot_until_index = len(x)
                    color_marker = 'green'
                    green_count += 1
                    green_dict_list.append({'BC': BC, 'M': M, 'Ech': Ech, 'Ep': Ep})

                # Definir a cor da trajetória
                last_x = x[plot_until_index - 1]
                if last_x < -0.0018 and color_marker != 'green':
                    color = 'red'
                    red_count += 1
                elif last_x > -0.0015 and color_marker != 'green':
                    color = 'blue'
                    blue_count += 1
                else:
                    black_count += 1

                # if color == 'red' or color == 'blue' and i >= len(x) - 1:
                #     color_marker = 'green'

                # Plotar a trajetória
                plt.plot(x[:plot_until_index], y[:plot_until_index], color=color)

                plt.scatter(x[plot_until_index - 1], y[plot_until_index - 1], color=color_marker, edgecolor='k', s=100, zorder=5)
                # print(f'{M}_{int(Ep)}')
                plt.text(x[plot_until_index - 1] * 0.99, y[plot_until_index - 1], f'{M}_{int(Ep)}', fontsize=10)

                if purple_marker is not None:
                    plt.scatter(x[purple_marker], y[purple_marker], color='purple', edgecolor='k', s=100, zorder=5)
                    purple_marker_count += 1

            # Título e etiquetas
            if BC == 1:
                plt.title(f'Trajectories for Diffusion BC, Ech={Ech}')
            else:
                plt.title(f'Trajectories for Standard BC, Ech={Ech}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True)
            
            # Legenda
            legend_labels = [
                f'Total: {total_count}',
                f'Red (x < -0.0018): {red_count}',
                f'Blue (x > -0.0015): {blue_count}',
                f'Black (Collision Event): {black_count}',
                f'Green (No Infractions): {green_count}',
                f'Purple Marker (Specific Infractions): {purple_marker_count}'
            ]
            print('Infrações:' + str(50 - (green_count)))
            [print(ele) for ele in green_dict_list]
            # print(green_dict_list)
            
            plt.tight_layout()
            plt.show()

# Caminho para os dados e chamada da função
# path_to_data_folder = "path_to_your_data_folder"
path_to_data_folder_2 = "diff_bc_video_(diffuser)/birdview/town01_multimodality_t_intersection_simples_2"
data = np.load(path_to_data_folder_2 + '/data_3.npy', allow_pickle=True)

plot_trajectories_by_bc_ep(data)
