import os
import re
import numpy as np
from collections import defaultdict

# Pasta que contém os arquivos
folder_path = "diff_bc_video_(diffuser)/birdview/new_arch_carla_route_plotter_3000_2/Diffusion_BC_Multi_Multiple_New_Arch_0_extra_steps/"

# Expressão regular mais flexível para extrair MODEL, EPISODE e INDEX
# Identifica o início "version_750_X" e o final "Y_Z_distance_traveled.txt"
pattern = r"version_750_(\d).*?_(\d+)_(\d+)_distance_traveled\.txt$"

# Dicionário para armazenar os valores de cada modelo e episódio
data = {
    0: defaultdict(list),
    1: defaultdict(list),
    2: defaultdict(list),
    3: defaultdict(list),
    4: defaultdict(list)
}

# Percorre todos os arquivos na pasta
for filename in os.listdir(folder_path):
    if filename.endswith("_distance_traveled.txt"):
        match = re.search(pattern, filename)
        if match:
            model = int(match.group(1))
            episode = int(match.group(2))
            index = int(match.group(3))
            
            # Lê o valor do arquivo
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    value = float(file.read().strip())
                    data[model][episode].append(value)
            except (IOError, ValueError) as e:
                print(f"Erro ao ler o arquivo {filename}: {e}")
        else:
            print(f"Arquivo ignorado (não corresponde ao padrão): {filename}")

# Calcula as médias para cada modelo e episódio
results = {}
for model in range(5):
    results[model] = {}
    for episode, values in data[model].items():
        if values:  # Verifica se há valores para calcular a média
            results[model][episode] = np.mean(values)

# Escreve os resultados no arquivo mean_distance.txt
with open("mean_distance.txt", 'w') as output_file:
    for model in range(5):
        output_file.write(f" MODEL {model}:\n")
        
        # Ordena os episódios por média em ordem decrescente
        sorted_episodes = sorted(results[model].items(), key=lambda x: x[1], reverse=True)
        
        for episode, mean_value in sorted_episodes:
            output_file.write(f"Episode {episode}: {mean_value:.2f}\n")
        
        # Adiciona uma linha em branco entre os modelos (exceto após o último)
        if model < 4:
            output_file.write("\n")

print("Arquivo mean_distance.txt criado com sucesso!")