import os
import re
import numpy as np
from collections import defaultdict

# Pasta que contém os arquivos
folder_path = "diff_bc_video_(diffuser)/birdview/Multiple_at_t/BC/BC_t_insersection_multiples_0_extra_steps/BC_t_insersection_multiples"

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
            model = 1
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
with open(folder_path+"/mean_distance.txt", 'w') as output_file:
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

def calculate_min_max():
    # Pasta que contém os arquivos
    folder_path = "diff_bc_video_(diffuser)/birdview/Multiple_at_t/BC/BC_t_insersection_multiples_0_extra_steps/BC_t_insersection_multiples/"
    # folder_path = "diff_bc_video_(diffuser)/birdview/Multiple_dbc_and_d2bc/Diffusion-BC/Diffusion_BC_Multi_Simple_New_Arch_2_0_extra_steps/version_750_4/"
    
    # Expressão regular mais flexível para extrair MODEL, EPISODE e INDEX
    # Identifica o início "version_750_X" e o final "Y_Z_distance_traveled.txt"
    pattern = r"version_750_(\d).*?_(\d+)_(\d+)_distance_traveled\.txt$"
    # pattern = r"new_arch_fixed_(\d).*?_(\d+)_(\d+)_distance_traveled\.txt$"

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
                # model = int(folder_path.split('/')[-2][-1])
                model = 1
                episode = int(match.group(2))
                index = int(match.group(3))
                
                # Verifica o arquivo correspondente sem "_distance_traveled"
                base_filename = filename.replace("_distance_traveled.txt", ".txt")
                base_file_path = os.path.join(folder_path, base_filename)
                
                # Verifica se o arquivo base existe e se atende ao critério
                if os.path.exists(base_file_path):
                    try:
                        # Carrega o array numpy do arquivo base
                        array_data = np.loadtxt(base_file_path)
                        
                        # Verifica se o último valor da primeira coluna é maior que 85
                        if array_data.shape[0] > 0 and array_data[-1, 0] > 85:
                            # Lê o valor do arquivo de distance_traveled
                            file_path = os.path.join(folder_path, filename)
                            try:
                                with open(file_path, 'r') as file:
                                    value = float(file.read().strip())
                                    data[model][episode].append(value)
                            except (IOError, ValueError) as e:
                                print(f"Erro ao ler o arquivo {filename}: {e}")
                        else:
                            print(f"Arquivo {base_filename} não atende ao critério (último valor da primeira coluna > 85)")
                    except Exception as e:
                        print(f"Erro ao processar o arquivo base {base_filename}: {e}")
                else:
                    print(f"Arquivo base não encontrado: {base_filename}")
            else:
                print(f"Arquivo ignorado (não corresponde ao padrão): {filename}")

    # Calcula os valores mínimos e máximos para cada modelo e episódio
    results = {}
    for model in range(5):
        results[model] = {}
        for episode, values in data[model].items():
            if values:  # Verifica se há valores para calcular min e max
                min_value = min(values)
                max_value = max(values)
                results[model][episode] = (min_value, max_value)

    # Escreve os resultados no arquivo min_max.txt
    with open(folder_path+"min_max.txt", 'w') as output_file:
        for model in range(5):
            output_file.write(f" MODEL {model}:\n")
            
            # Ordena os episódios pela média dos valores min e max em ordem decrescente
            # para manter a mesma ordem do arquivo original
            sorted_episodes = sorted(
                results[model].items(), 
                key=lambda x: (x[1][0] + x[1][1]) / 2, 
                reverse=True
            )
            
            for episode, (min_value, max_value) in sorted_episodes:
                output_file.write(f"Episode {episode}: ({min_value:.2f}, {max_value:.2f})\n")
            
            # Adiciona uma linha em branco entre os modelos (exceto após o último)
            if model < 4:
                output_file.write("\n")

    print("Arquivo min_max.txt criado com sucesso!")


calculate_min_max()

def create_top5_histograms_with_min_max(folder_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import re

    # Fonte grande global para todos os elementos
    plt.rcParams.update({'font.size': 18})

    # Caminho para os arquivos de média e min_max
    mean_file = os.path.join(folder_path, "mean_distance.txt")
    min_max_file = os.path.join(folder_path, "min_max.txt")
    
    # Dicionários para armazenar os dados lidos
    mean_data = {i: {} for i in range(5)}  # {modelo: {episódio: média}}
    min_max_data = {i: {} for i in range(5)}  # {modelo: {episódio: (min, max)}}
    
    # Lê o arquivo de médias
    current_model = None
    with open(mean_file, 'r') as file:
        for line in file:
            line = line.strip()
            model_match = re.match(r'\s*MODEL\s+(\d):', line)
            if model_match:
                current_model = int(model_match.group(1))
                continue
            episode_match = re.match(r'Episode\s+(\d+):\s+([\d.]+)', line)
            if episode_match and current_model is not None:
                episode = int(episode_match.group(1))
                mean_value = float(episode_match.group(2))
                mean_data[current_model][episode] = mean_value
    
    # Lê o arquivo de min_max
    current_model = None
    with open(min_max_file, 'r') as file:
        for line in file:
            line = line.strip()
            model_match = re.match(r'\s*MODEL\s+(\d):', line)
            if model_match:
                current_model = int(model_match.group(1))
                continue
            episode_match = re.match(r'Episode\s+(\d+):\s+\(([\d.]+),\s*([\d.]+)\)', line)
            if episode_match and current_model is not None:
                episode = int(episode_match.group(1))
                min_value = float(episode_match.group(2))
                max_value = float(episode_match.group(3))
                min_max_data[current_model][episode] = (min_value, max_value)
    
    for model in range(5):
        sorted_episodes = sorted(mean_data[model].items(), key=lambda x: x[1], reverse=True)
        top5_episodes = sorted_episodes[:5]
        episodes = [ep for ep, _ in top5_episodes]
        episodes.sort()
        
        means = [mean_data[model][ep] for ep in episodes]
        min_values = [min_max_data[model].get(ep, (0, 0))[0] for ep in episodes]
        max_values = [min_max_data[model].get(ep, (0, 0))[1] for ep in episodes]
        
        yerr_low = [means[i] - min_values[i] for i in range(len(episodes))]
        yerr_high = [max_values[i] - means[i] for i in range(len(episodes))]
        yerr = [yerr_low, yerr_high]
        
        plt.figure(figsize=(10, 6))
        x_indices = np.arange(len(episodes))
        
        bars = plt.bar(x_indices, means, width=0.7, color='blue', alpha=0.9)
        plt.errorbar(x_indices, means, yerr=yerr, fmt='none', ecolor='red', capsize=10, capthick=2, elinewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.ylim(-100000, 400000)
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Average Distance', fontsize=20)
        # plt.title(f'Top 5 Episódios - Modelo {model} (com Min/Max)', fontsize=22)
        
        plt.xticks(x_indices, [str(ep) for ep in episodes], fontsize=18)
        plt.yticks(fontsize=18)
        
        for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
            plt.annotate(f'Min: {min_val:.0f}', 
                         xy=(x_indices[i], min_val), 
                         xytext=(x_indices[i] - 0.25, min_val - 30000),
                         fontsize=14)
            plt.annotate(f'Max: {max_val:.0f}', 
                         xy=(x_indices[i], max_val), 
                         xytext=(x_indices[i] - 0.25, max_val + 10000),
                         fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f"histogram_min_max_{model}.png"), dpi=300)
        plt.close()
        
        print(f"Histograma do modelo {model} com os 5 melhores episódios (min/max) salvo com sucesso!")


folder_path = "diff_bc_video_(diffuser)/birdview/Multiple_at_t/BC/BC_t_insersection_multiples_0_extra_steps/BC_t_insersection_multiples/"
create_top5_histograms_with_min_max(folder_path)