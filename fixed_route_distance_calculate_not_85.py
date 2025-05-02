import os
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import statistics
plt.rcParams.update({
    'font.size': 18,          # dobra o tamanho padrão (~10 → 22)
    'axes.labelsize': 18,     # tamanho do xlabel e ylabel
    'axes.titlesize': 18,     # se você usar plt.title
    'xtick.labelsize': 18,    # tamanho dos valores no eixo X
    'ytick.labelsize': 18,    # tamanho dos valores no eixo Y
    'legend.fontsize': 18,    # legenda
})

def compute_epoch_means_from_csv(folder_path, output_filename="median_curve.txt"):
    """
    Lê arquivos .csv de uma pasta, agrupa os valores da coluna 'route_completed_in_m' por EPOCH,
    calcula a média para cada EPOCH (desconsiderando arquivos com menos de 5 linhas),
    e salva em um arquivo de texto.

    Args:
        folder_path (str): Caminho para a pasta com arquivos .csv
        output_filename (str): Nome do arquivo de saída (padrão: 'median_curve.txt')

    Returns:
        None
    """
    epoch_values = defaultdict(list)
    len_value = defaultdict(list)

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and "ep_" in filename:
            try:
                parts = filename.split("_")
                epoch_index = parts.index("ep") + 1
                epoch = int(parts[epoch_index])
            except (ValueError, IndexError):
                continue  # pula arquivos com nomes fora do padrão

            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Erro ao ler {file_path}: {e}")
                continue
            
            df = df.iloc[:3001]
            if len(df) < 5:
                continue

            if "route_completed_in_m" in df.columns:
                mean_value = df["route_completed_in_m"].values[-1]
                epoch_values[epoch].append(mean_value)
                len_value[epoch].append(len(df))
    epoch_std_devs = {
        epoch: statistics.stdev(values)
        for epoch, values in epoch_values.items()
    }

    epoch_averages = {
        epoch: sum(values) / len(values)
        for epoch, values in epoch_values.items()
    }

    median_completion = {
        epoch: statistics.mean(values)/3000
        for epoch, values in len_value.items()
    }

    std_completion = {
        epoch: statistics.stdev([ele/ 3000 for ele in values])
        for epoch, values in len_value.items()
    }

    # Salva no arquivo
    output_path = os.path.join(folder_path, output_filename)
    with open(output_path, "w") as f:
        for epoch in sorted(epoch_averages):
            f.write(f"{epoch} {epoch_averages[epoch]:.4f}\n")

    print(f"Arquivo '{output_filename}' salvo em: {output_path}")



    output_path = os.path.join(folder_path, '0std_dev_curve.txt')
    with open(output_path, "w") as f:
        for epoch in sorted(epoch_std_devs):
            f.write(f"{epoch} {epoch_std_devs[epoch]:.4f}\n")

    print(f"Arquivo '{'0std_dev_curve.txt'}' salvo em: {output_path}")


    output_path = os.path.join(folder_path, 'median_completion.txt')
    with open(output_path, "w") as f:
        for epoch in sorted(median_completion):
            f.write(f"{epoch} {median_completion[epoch]:.4f}\n")

    print(f"Arquivo '{output_filename}' salvo em: {output_path}")



    output_path = os.path.join(folder_path, '0std_dev_completion.txt')
    with open(output_path, "w") as f:
        for epoch in sorted(std_completion):
            f.write(f"{epoch} {std_completion[epoch]:.4f}\n")

    print(f"Arquivo '{'0std_dev_curve.txt'}' salvo em: {output_path}")


def compute_general_median_curve(base_path):
    # Armazenar os valores por EPOCH
    epoch_values = defaultdict(list)

    # Iterar sobre VERSIONs 0, 1, 2
    for version in range(3):
        median_curve_path = os.path.join(
            base_path,
            f"BC_Full_Trajectory_300_0{version}_0_extra_steps",
            f"BC_Full_Trajectory_300_0{version}",
            "median_curve.txt"
        )

        # Verifica se o arquivo existe
        if not os.path.exists(median_curve_path):
            print(f"Aviso: {median_curve_path} não encontrado.")
            continue

        # Lê os dados do arquivo
        with open(median_curve_path, "r") as f:
            for line in f:
                try:
                    epoch, value = line.strip().split()
                    epoch = int(epoch)
                    value = float(value)
                    epoch_values[epoch].append(value)
                except ValueError:
                    continue  # ignora linhas mal formatadas

    # Calcula média e desvio padrão para cada EPOCH
    output_lines = []
    for epoch in sorted(epoch_values):
        values = np.array(epoch_values[epoch])
        mean = values.mean()
        std = values.std()
        output_lines.append(f"{epoch} {mean:.4f} {std:.4f}")

    # Salva no arquivo final
    output_path = os.path.join(base_path, "median_general_curve.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Arquivo 'median_general_curve.txt' salvo em: {output_path}")


def plot_median_general_curve(file_path, output_path="median_general_curve_plot.png"):
    """
    Plota a curva de performance (média ± desvio padrão) a partir do arquivo median_general_curve.txt

    Args:
        file_path (str): Caminho para o arquivo 'median_general_curve.txt'
        output_path (str): Caminho do arquivo de imagem a ser salvo (formato PNG)

    Returns:
        None
    """
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return

    # Listas para armazenar os dados
    epochs, means, stds = [], [], []

    # Lê o arquivo
    with open(file_path, "r") as f:
        for line in f:
            try:
                epoch, mean, std = line.strip().split()
                epochs.append(int(epoch))
                means.append(float(mean))
                stds.append(float(std))
            except ValueError:
                continue  # Ignora linhas mal formatadas

    # Converte para numpy
    epochs = np.array(epochs)
    means = np.array(means)
    stds = np.array(stds)

    # Cria o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, means, label="BC Full Trajectory", color="orange")
    plt.fill_between(epochs, means - stds, means + stds, color="orange", alpha=0.3)

    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("Mean Route Distance in m", fontsize=12)
    # plt.title("Performance over training (mean ± std)", fontsize=14)
    # plt.legend()
    plt.ylim((0, 700))
    plt.grid(True)
    plt.tight_layout()

    # Salva ou exibe
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico salvo em: {output_path}")

import os
import numpy as np
from collections import defaultdict

def compute_general_median_curve_diffusion(model_number):
    """
    Calcula a média e o desvio padrão dos valores de 'median_curve.txt' para diferentes versões,
    considerando um modelo específico.

    Args:
        model_number (int): Número do modelo a ser considerado (presente no nome da pasta)

    Salva:
        Um arquivo 'median_general_curve_{MODEL}.txt' com epoch, média e desvio padrão.
    """
    base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    version_template = "Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_{}_0_extra_steps/version_750_{}"
    num_versions = 3  # versões 0, 1 e 2

    # Armazena os valores por EPOCH
    epoch_data = defaultdict(list)

    for version in range(num_versions):
        if version == 0:
            continue
        folder = os.path.join(base_path, version_template.format(version, model_number))
        file_path = os.path.join(folder, "median_curve.txt")

        if not os.path.isfile(file_path):
            print(f"Aviso: '{file_path}' não encontrado. Pulando.")
            continue

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                try:
                    epoch = int(parts[0])
                    value = float(parts[1])
                    epoch_data[epoch].append(value)
                except ValueError:
                    continue

    # Calcula média e desvio padrão por EPOCH
    stats_by_epoch = {
        epoch: (np.mean(values), np.std(values))
        for epoch, values in epoch_data.items()
    }

    # Salva no arquivo
    output_path = os.path.join(base_path, f"median_general_curve_{model_number}.txt")
    with open(output_path, "w") as f:
        for epoch in sorted(stats_by_epoch):
            mean, std = stats_by_epoch[epoch]
            f.write(f"{epoch} {mean:.4f} {std:.4f}\n")

    print(f"Arquivo 'median_general_curve_{model_number}.txt' salvo em: {output_path}")


import os
import numpy as np
from collections import defaultdict

def compute_general_median_curve_diffusion(model_number):
    """
    Calcula a média e o desvio padrão dos valores de 'median_curve.txt' para diferentes versões
    de um modelo específico dentro da estrutura Diffusion-BC.

    Args:
        model_number (int): Número do modelo (ex: 39, 42, etc.)

    Salva:
        Arquivo 'median_general_curve_{MODEL}.txt' com média e desvio padrão por EPOCH.
    """
    base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    version_folder_template = "Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_{}_0_extra_steps/version_750_{}"
    output_filename = f"median_general_curve_{model_number}.txt"

    epoch_data = defaultdict(list)

    for version in range(3):  # versões 0, 1, 2
        if version == 0:
            continue
        version_path = os.path.join(base_path, version_folder_template.format(version, model_number))
        median_curve_path = os.path.join(version_path, "median_curve.txt")

        if not os.path.exists(median_curve_path):
            print(f"Aviso: arquivo não encontrado para versão {version}: {median_curve_path}")
            continue

        with open(median_curve_path, "r") as file:
            for line in file:
                try:
                    epoch, value = line.strip().split()
                    epoch = int(epoch)
                    value = float(value)
                    epoch_data[epoch].append(value)
                except ValueError:
                    continue  # pula linhas com erro

    # Calcula média e desvio padrão para cada EPOCH
    stats_by_epoch = {
        epoch: (np.mean(vals), np.std(vals)) for epoch, vals in epoch_data.items()
    }

    # Salva no arquivo
    output_path = os.path.join(base_path, output_filename)
    with open(output_path, "w") as f:
        for epoch in sorted(stats_by_epoch):
            mean, std = stats_by_epoch[epoch]
            f.write(f"{epoch} {mean:.4f} {std:.4f}\n")

    print(f"Arquivo '{output_filename}' salvo em: {output_path}")


import os


def plot_median_general_curve_diffusion(model_number):
    """
    Plota o gráfico de média e desvio padrão por EPOCH a partir do arquivo
    'median_general_curve_{MODEL}.txt'.

    Args:
        model_number (int): número do modelo a ser carregado
    """
    base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    filename = f"median_general_curve_{model_number}.txt"
    file_path = os.path.join(base_path, filename)

    if not os.path.exists(file_path):
        print(f"Arquivo '{filename}' não encontrado em {base_path}")
        return

    # Carrega os dados
    epochs, means, stds = [], [], []
    with open(file_path, "r") as f:
        for line in f:
            try:
                epoch, mean, std = line.strip().split()
                epochs.append(int(epoch))
                means.append(float(mean))
                stds.append(float(std))
            except ValueError:
                continue

    # Ordena por epoch
    sorted_data = sorted(zip(epochs, means, stds), key=lambda x: x[0])
    epochs, means, stds = zip(*sorted_data)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, means, label="Média por EPOCH", color="blue")
    plt.fill_between(epochs,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     color="blue", alpha=0.2, label="±1 desvio padrão")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Route Distance in m")
    # plt.title(f"Média e Desvio Padrão - Modelo {model_number}")
    # plt.legend()
    plt.ylim((0, 700))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    output_img = os.path.join(base_path, f"median_general_curve_{model_number}.png")
    plt.savefig(output_img)
    print(f"Figura salva em: {output_img}")


import os
import pandas as pd
from collections import defaultdict

def compute_relative_completion_means_from_csv(folder_path):
    """
    Calcula a média de (último route_completed_in_m / route_length_in_m)
    por EPOCH em arquivos CSV no formato ep_{EPOCH}_{EPISODE}.csv.
    
    Regras:
      - Ignora arquivos com menos de 5 linhas.
      - Se tiver mais de 2999 linhas, considera valor como 1.
      - Usa última linha para pegar valores de route_completed_in_m / route_length_in_m.
    
    Salva resultado em 'relative_completion_curve.txt' no mesmo diretório.
    """

    epoch_values = defaultdict(list)

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and "ep_" in filename:
            try:
                parts = filename.split("_")
                epoch_index = parts.index("ep") + 1
                epoch = int(parts[epoch_index])
            except (ValueError, IndexError):
                continue  # ignora arquivos fora do padrão

            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if len(df) < 5:
                continue  # ignora arquivos curtos

            if len(df) > 2999:
                ratio = 1.0
            else:
                if "route_completed_in_m" in df.columns and "route_length_in_m" in df.columns:
                    completed = df["route_completed_in_m"].iloc[-1]
                    length = df["route_length_in_m"].iloc[-1]
                    if length > 0:
                        ratio = completed / length
                    else:
                        continue  # evita divisão por zero
                else:
                    continue  # ignora arquivos que não têm as colunas

            epoch_values[epoch].append(ratio)

    # Calcula média por EPOCH
    epoch_averages = {
        epoch: sum(values) / len(values)
        for epoch, values in epoch_values.items()
    }

    # Salva arquivo
    output_path = os.path.join(folder_path, "relative_completion_curve.txt")
    with open(output_path, "w") as f:
        for epoch in sorted(epoch_averages):
            f.write(f"{epoch} {epoch_averages[epoch]:.4f}\n")

    print(f"Arquivo 'relative_completion_curve.txt' salvo em: {output_path}")


import os
import numpy as np
from collections import defaultdict

def compute_general_relative_completion_curve():
    """
    Agrupa os valores de relative_completion_curve.txt de cada versão (0, 1, 2),
    calcula média e desvio padrão por EPOCH, e salva em:
    'diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/relative_completion_general_curve.txt'
    """

    base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC"
    versions = [0, 1, 2]
    epoch_values = defaultdict(list)

    for version in versions:
        folder = f"BC_Full_Trajectory_300_0{version}_0_extra_steps/BC_Full_Trajectory_300_0{version}"
        file_path = os.path.join(base_path, folder, "relative_completion_curve.txt")

        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            continue

        with open(file_path, "r") as f:
            for line in f:
                try:
                    epoch, value = line.strip().split()
                    epoch = int(epoch)
                    value = float(value)
                    epoch_values[epoch].append(value)
                except ValueError:
                    continue  # ignora linhas mal formatadas

    # Calcula média e desvio padrão
    output_lines = []
    for epoch in sorted(epoch_values):
        values = epoch_values[epoch]
        mean = np.mean(values)
        std = np.std(values)
        output_lines.append(f"{epoch} {mean:.4f} {std:.4f}")

    # Salva resultado
    output_path = os.path.join(base_path, "relative_completion_general_curve.txt")
    with open(output_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"Arquivo salvo em: {output_path}")


import os

def plot_relative_completion_general_curve():
    """
    Plota o gráfico da curva média de completion relativa com desvio padrão.
    Lê 'relative_completion_general_curve.txt' e salva imagem como .png.
    """

    input_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/relative_completion_general_curve.txt"
    output_img = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/relative_completion_general_curve.png"

    epochs = []
    means = []
    stds = []

    if not os.path.exists(input_path):
        print(f"Arquivo não encontrado: {input_path}")
        return

    with open(input_path, "r") as f:
        for line in f:
            try:
                epoch, mean, std = line.strip().split()
                epochs.append(int(epoch))
                means.append(float(mean))
                stds.append(float(std))
            except ValueError:
                continue

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, means, label="Mean", color="blue")
    plt.fill_between(epochs,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     color="blue", alpha=0.3, label="±1 Standard Deviation")

    plt.xlabel("Epoch")
    plt.ylabel("Mean Relative Completion")
    # plt.title("Curva de Completion Relativo com Desvio Padrão")
    plt.grid(True)
    # plt.legend()
    plt.ylim((-0.1, 1.1))
    plt.tight_layout()

    plt.savefig(output_img)
    plt.close()

    print(f"Gráfico salvo em: {output_img}")


import os
import pandas as pd
from collections import defaultdict

def compute_relative_completion_means_from_csv_diffusion(model):
    base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    for version in range(3):
        folder_path = os.path.join(
            base_path,
            f"Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_{version}_0_extra_steps",
            f"version_750_{model}"
        )

        if not os.path.exists(folder_path):
            print(f"Pasta não encontrada: {folder_path}")
            continue

        epoch_values = defaultdict(list)

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv") and "ep_" in filename:
                try:
                    parts = filename.split("_")
                    epoch_index = parts.index("ep") + 1
                    epoch = int(parts[epoch_index])
                except (ValueError, IndexError):
                    continue

                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)

                if len(df) < 5:
                    continue

                if len(df) >= 3000:
                    value = 1.0
                elif "route_completed_in_m" in df.columns and "route_length_in_m" in df.columns:
                    route_completed = df["route_completed_in_m"].iloc[-1]
                    route_length = df["route_length_in_m"].iloc[-1]
                    if route_length > 0:
                        value = route_completed / route_length
                    else:
                        continue
                else:
                    continue

                epoch_values[epoch].append(value)

        # Calcula médias e salva
        epoch_averages = {
            epoch: sum(values) / len(values)
            for epoch, values in epoch_values.items()
        }

        output_path = os.path.join(folder_path, "diffusion_relative_completion_curve.txt")
        with open(output_path, "w") as f:
            for epoch in sorted(epoch_averages):
                f.write(f"{epoch} {epoch_averages[epoch]:.4f}\n")

        print(f"[VERSION {version}] Arquivo salvo: {output_path}")



import numpy as np

def compute_general_relative_completion_curve_diffusion(model):
    base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    versions = [0, 1, 2]
    epoch_values = defaultdict(list)

    for version in versions:
        folder = f"Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_{version}_0_extra_steps/version_750_{model}"
        file_path = os.path.join(base_path, folder, "diffusion_relative_completion_curve.txt")

        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            continue

        with open(file_path, "r") as f:
            for line in f:
                try:
                    epoch, value = line.strip().split()
                    epoch = int(epoch)
                    value = float(value)
                    epoch_values[epoch].append(value)
                except ValueError:
                    continue

    output_lines = []
    for epoch in sorted(epoch_values):
        values = epoch_values[epoch]
        mean = np.mean(values)
        std = np.std(values)
        output_lines.append(f"{epoch} {mean:.4f} {std:.4f}")

    output_path = os.path.join(base_path, f"relative_completion_general_curve_{model}.txt")
    with open(output_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"Arquivo geral salvo: {output_path}")



def plot_relative_completion_general_curve_diffusion(model):
    base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    input_path = os.path.join(base_path, f"relative_completion_general_curve_{model}.txt")
    output_img = os.path.join(base_path, f"relative_completion_general_curve_{model}.png")

    if not os.path.exists(input_path):
        print(f"Arquivo não encontrado: {input_path}")
        return

    epochs, means, stds = [], [], []

    with open(input_path, "r") as f:
        for line in f:
            try:
                epoch, mean, std = line.strip().split()
                epochs.append(int(epoch))
                means.append(float(mean))
                stds.append(float(std))
            except ValueError:
                continue

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, means, label="Mean", color="green")
    plt.fill_between(epochs,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     color="green", alpha=0.3, label="±1 Standard Deviation")

    plt.xlabel("Epoch")
    plt.ylabel("Mean Relative Completion")
    # plt.title(f"Curva de Completion Relativo com Desvio Padrão (Modelo {model})")
    plt.grid(True)
    plt.ylim((-0.1, 1.1))
    # plt.legend()
    plt.tight_layout()

    plt.savefig(output_img)
    plt.close()

    print(f"Gráfico salvo em: {output_img}")


def plot_median_general_curve_diffusion_with_bc(model_number):
    """
    Plota o gráfico de média e desvio padrão por EPOCH do modelo Diffusion-BC e do modelo BC.

    Args:
        model_number (int): número do modelo a ser carregado
    """
    diffusion_base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    diffusion_filename = f"median_general_curve_{model_number}.txt"
    diffusion_file_path = os.path.join(diffusion_base_path, diffusion_filename)

    bc_file_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/median_general_curve.txt"

    if not os.path.exists(diffusion_file_path):
        print(f"Arquivo '{diffusion_filename}' não encontrado em {diffusion_base_path}")
        return

    # Carrega os dados do Diffusion-BC
    diffusion_epochs, diffusion_means, diffusion_stds = [], [], []
    with open(diffusion_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                epoch, mean, std = parts
                diffusion_epochs.append(int(epoch))
                diffusion_means.append(float(mean))
                diffusion_stds.append(float(std))

    if diffusion_epochs:
        sorted_diff = sorted(zip(diffusion_epochs, diffusion_means, diffusion_stds), key=lambda x: x[0])
        diffusion_epochs, diffusion_means, diffusion_stds = zip(*sorted_diff)

    # Carrega os dados do BC
    bc_epochs, bc_means, bc_stds = [], [], []
    if os.path.exists(bc_file_path):
        with open(bc_file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    epoch, mean, std = parts
                    bc_epochs.append(int(epoch))
                    bc_means.append(float(mean))
                    bc_stds.append(float(std))
        if bc_epochs:
            sorted_bc = sorted(zip(bc_epochs, bc_means, bc_stds), key=lambda x: x[0])
            bc_epochs, bc_means, bc_stds = zip(*sorted_bc)
    else:
        print(f"Arquivo de BC não encontrado em {bc_file_path}")

    # Plot
    plt.figure(figsize=(10, 6))

    # Diffusion-BC
    if diffusion_epochs:
        plt.plot(diffusion_epochs, diffusion_means, label="Diffusion-BC", color="blue")
        plt.fill_between(diffusion_epochs,
                         [m - s for m, s in zip(diffusion_means, diffusion_stds)],
                         [m + s for m, s in zip(diffusion_means, diffusion_stds)],
                         color="blue", alpha=0.2)

    # BC
    if bc_epochs:
        plt.plot(bc_epochs, bc_means, label="MSE-BC", color="orange")
        plt.fill_between(bc_epochs,
                         [m - s for m, s in zip(bc_means, bc_stds)],
                         [m + s for m, s in zip(bc_means, bc_stds)],
                         color="orange", alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Route Distance in m")
    plt.ylim((0, 700))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Salva a figura
    output_img = os.path.join(diffusion_base_path, f"median_general_curve_{model_number}_with_bc.png")
    plt.savefig(output_img)
    print(f"Figura salva em: {output_img}")



def plot_relative_completion_general_curve_diffusion_with_bc(model):
    base_path_diff = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    input_path_diff = os.path.join(base_path_diff, f"relative_completion_general_curve_{model}.txt")
    output_img = os.path.join(base_path_diff, f"relative_completion_general_curve_{model}_with_bc.png")

    base_path_bc = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC"
    input_path_bc = os.path.join(base_path_bc, "median_general_curve.txt")

    if not os.path.exists(input_path_diff):
        print(f"Arquivo não encontrado: {input_path_diff}")
        return

    # Leitura dos dados Diffusion-BC
    epochs_diff, means_diff, stds_diff = [], [], []
    with open(input_path_diff, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                epoch, mean, std = parts
                epochs_diff.append(int(epoch))
                means_diff.append(float(mean))
                stds_diff.append(float(std))

    # Leitura dos dados do BC
    epochs_bc, means_bc, stds_bc = [], [], []
    if os.path.exists(input_path_bc):
        with open(input_path_bc, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    epoch, mean, std = parts
                    epochs_bc.append(int(epoch))
                    means_bc.append(float(mean) / 700.0)  # assumindo 700m como total para normalização
                    stds_bc.append(float(std) / 700.0)
    else:
        print(f"Arquivo BC não encontrado: {input_path_bc}")

    # Plot
    plt.figure(figsize=(10, 6))

    # Diffusion-BC
    if epochs_diff:
        plt.plot(epochs_diff, means_diff, label="Diffusion-BC", color="blue")
        plt.fill_between(epochs_diff,
                         [m - s for m, s in zip(means_diff, stds_diff)],
                         [m + s for m, s in zip(means_diff, stds_diff)],
                         color="blue", alpha=0.3)

    # BC
    if epochs_bc:
        plt.plot(epochs_bc, means_bc, label="MSE-BC", color="orange")
        plt.fill_between(epochs_bc,
                         [m - s for m, s in zip(means_bc, stds_bc)],
                         [m + s for m, s in zip(means_bc, stds_bc)],
                         color="orange", alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Relative Completion")
    plt.ylim((-0.1, 1.1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()

    print(f"Gráfico salvo em: {output_img}")

import os
import matplotlib.pyplot as plt

def plot_relative_completion_general_curve_diffusion_with_bc_2bc(model):
    base_path_diff = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    input_path_diff = os.path.join(base_path_diff, f"relative_completion_general_curve_{model}.txt")
    input_path_diff0 = os.path.join(base_path_diff, "relative_completion_general_curve_0.txt")
    output_img = os.path.join(base_path_diff, f"relative_completion_general_curve_{model}_with_bc_2bc.png")

    base_path_bc = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC"
    input_path_bc = os.path.join(base_path_bc, "median_general_curve.txt")

    def read_curve_file(path, normalize=False):
        epochs, means, stds = [], [], []
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        epoch, mean, std = parts
                        mean_val = float(mean)
                        std_val = float(std)
                        if normalize:
                            mean_val /= 700.0
                            std_val /= 700.0
                        epochs.append(int(epoch))
                        means.append(mean_val)
                        stds.append(std_val)
        else:
            print(f"Arquivo não encontrado: {path}")
        return epochs, means, stds

    # Leitura dos arquivos
    epochs_diff, means_diff, stds_diff = read_curve_file(input_path_diff)
    epochs_diff0, means_diff0, stds_diff0 = read_curve_file(input_path_diff0)
    epochs_bc, means_bc, stds_bc = read_curve_file(input_path_bc, normalize=True)

    # Plot
    plt.figure(figsize=(10, 6))

    if epochs_diff:
        plt.plot(epochs_diff, means_diff, label="Diffusion-2BC", color="green")
        plt.fill_between(epochs_diff,
                         [m - s for m, s in zip(means_diff, stds_diff)],
                         [m + s for m, s in zip(means_diff, stds_diff)],
                         color="green", alpha=0.3)

    if epochs_diff0:
        plt.plot(epochs_diff0, means_diff0, label="Diffusion-BC", color="blue")
        plt.fill_between(epochs_diff0,
                         [m - s for m, s in zip(means_diff0, stds_diff0)],
                         [m + s for m, s in zip(means_diff0, stds_diff0)],
                         color="blue", alpha=0.2)

    if epochs_bc:
        plt.plot(epochs_bc, means_bc, label="MSE-BC", color="orange")
        plt.fill_between(epochs_bc,
                         [m - s for m, s in zip(means_bc, stds_bc)],
                         [m + s for m, s in zip(means_bc, stds_bc)],
                         color="orange", alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Relative Completion")
    plt.ylim((-0.1, 1.1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()

    print(f"Gráfico salvo em: {output_img}")


import os
import matplotlib.pyplot as plt

def plot_median_general_curve_diffusion_with_bc_2bc(model_number):
    """
    Plota o gráfico de média e desvio padrão por EPOCH do modelo Diffusion-BC, Diffusion-BC (sem fine-tuning) e do modelo BC.

    Args:
        model_number (int): número do modelo a ser carregado
    """
    diffusion_base_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC"
    diffusion_file_path = os.path.join(diffusion_base_path, f"median_general_curve_{model_number}.txt")
    diffusion_file_path_0 = os.path.join(diffusion_base_path, "median_general_curve_0.txt")
    bc_file_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/median_general_curve.txt"

    def read_curve_file(path):
        epochs, means, stds = [], [], []
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        epoch, mean, std = parts
                        epochs.append(int(epoch))
                        means.append(float(mean))
                        stds.append(float(std))
            if epochs:
                sorted_data = sorted(zip(epochs, means, stds), key=lambda x: x[0])
                epochs, means, stds = zip(*sorted_data)
        else:
            print(f"Arquivo não encontrado: {path}")
        return epochs, means, stds

    # Leitura dos dados
    diffusion_epochs, diffusion_means, diffusion_stds = read_curve_file(diffusion_file_path)
    diffusion_epochs_0, diffusion_means_0, diffusion_stds_0 = read_curve_file(diffusion_file_path_0)
    bc_epochs, bc_means, bc_stds = read_curve_file(bc_file_path)

    # Plot
    plt.figure(figsize=(10, 6))

    if diffusion_epochs:
        plt.plot(diffusion_epochs, diffusion_means, label="Diffusion-2BC", color="green")
        plt.fill_between(diffusion_epochs,
                         [m - s for m, s in zip(diffusion_means, diffusion_stds)],
                         [m + s for m, s in zip(diffusion_means, diffusion_stds)],
                         color="green", alpha=0.2)

    if diffusion_epochs_0:
        plt.plot(diffusion_epochs_0, diffusion_means_0, label="Diffusion-BC", color="blue")
        plt.fill_between(diffusion_epochs_0,
                         [m - s for m, s in zip(diffusion_means_0, diffusion_stds_0)],
                         [m + s for m, s in zip(diffusion_means_0, diffusion_stds_0)],
                         color="blue", alpha=0.2)

    if bc_epochs:
        plt.plot(bc_epochs, bc_means, label="MSE-BC", color="orange")
        plt.fill_between(bc_epochs,
                         [m - s for m, s in zip(bc_means, bc_stds)],
                         [m + s for m, s in zip(bc_means, bc_stds)],
                         color="orange", alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Mean Route Distance in m")
    plt.ylim((0, 700))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Salva a figura
    output_img = os.path.join(diffusion_base_path, f"median_general_curve_{model_number}_with_bc_2bc.png")
    plt.savefig(output_img)
    plt.close()
    print(f"Figura salva em: {output_img}")



if __name__ == '__main__':

    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_3/BC/BC_Full_Trajectory_300_02_0_extra_steps/BC_Full_Trajectory_300_02"
    )

    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_3/BC/BC_Full_Trajectory_300_01_0_extra_steps/BC_Full_Trajectory_300_01"
    )

    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_3/BC/BC_Full_Trajectory_300_00_0_extra_steps/BC_Full_Trajectory_300_00"
    )











    plot_relative_completion_general_curve_diffusion_with_bc_2bc(2)
    plot_median_general_curve_diffusion_with_bc_2bc(2)

    compute_relative_completion_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_00_0_extra_steps/BC_Full_Trajectory_300_00"
    )
    compute_relative_completion_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_01_0_extra_steps/BC_Full_Trajectory_300_01"
    )
    compute_relative_completion_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_02_0_extra_steps/BC_Full_Trajectory_300_02"
    )
    compute_relative_completion_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_00_0_extra_steps/BC_Full_Trajectory_300_00"
    )
    compute_relative_completion_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_01_0_extra_steps/BC_Full_Trajectory_300_01"
    )
    compute_relative_completion_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_02_0_extra_steps/BC_Full_Trajectory_300_02"
    )
    compute_general_relative_completion_curve()
    plot_relative_completion_general_curve()

    compute_relative_completion_means_from_csv_diffusion(0)
    compute_general_relative_completion_curve_diffusion(0)
    plot_relative_completion_general_curve_diffusion(0)

    compute_relative_completion_means_from_csv_diffusion(2)
    compute_general_relative_completion_curve_diffusion(2)
    plot_relative_completion_general_curve_diffusion(2)




    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_00_0_extra_steps/BC_Full_Trajectory_300_00"
    )
    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_01_0_extra_steps/BC_Full_Trajectory_300_01"
    )
    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/BC_Full_Trajectory_300_02_0_extra_steps/BC_Full_Trajectory_300_02"
    )
    base_bc_path = "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC"
    compute_general_median_curve(base_bc_path)
    plot_median_general_curve(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/median_general_curve.txt",
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/BC/median_general_curve_plot.png"
    )

    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_0_0_extra_steps/version_750_0"
    )
    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_0_0_extra_steps/version_750_2"
    )
    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_1_0_extra_steps/version_750_0"
    )
    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_1_0_extra_steps/version_750_2"
    )
    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_2_0_extra_steps/version_750_0"
    )
    compute_epoch_means_from_csv(
        "diff_bc_video_(diffuser)/birdview/Fixed_Route_2/Diffusion-BC/Diffusion_BC_Multi_Fixed_New_Arch_Full_1ep_2_0_extra_steps/version_750_2"
    )
    compute_general_median_curve_diffusion(0)
    plot_median_general_curve_diffusion(0)
    compute_general_median_curve_diffusion(2)
    plot_median_general_curve_diffusion(2)

    plot_median_general_curve_diffusion_with_bc(0)
    plot_relative_completion_general_curve_diffusion_with_bc(0)

    plot_median_general_curve_diffusion_with_bc(2)
    plot_relative_completion_general_curve_diffusion_with_bc(2)