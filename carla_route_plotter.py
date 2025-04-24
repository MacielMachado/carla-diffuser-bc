import queue
from pathlib import Path
import ast
from PIL import Image, ImageDraw
import numpy as np
import carla
import random
from typing import List, Tuple, Union, Dict, Optional

import queue
from pathlib import Path
import ast  # For safely evaluating string representation of lists

from PIL import Image, ImageDraw, ImageColor
import numpy as np

import carla
from carla import ColorConverter as cc

from carla_gym.utils import config_utils
from carla_gym.core.task_actor.common.navigation.global_route_planner import GlobalRoutePlanner
from carla_gym.core.task_actor.common.navigation.route_manipulation import downsample_route
import random
from carla_gym.core.task_actor.common.navigation.global_route_planner import RoadOption

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams.update({
    'font.size': 22,          # dobra o tamanho padrão (~10 → 22)
    'axes.labelsize': 22,     # tamanho do xlabel e ylabel
    'axes.titlesize': 22,     # se você usar plt.title
    'xtick.labelsize': 22,    # tamanho dos valores no eixo X
    'ytick.labelsize': 22,    # tamanho dos valores no eixo Y
    'legend.fontsize': 22,    # legenda
})


class CarlaRoutePlotter:
    def __init__(self, host: str = 'localhost', port: int = 2020, town: str = 'Town01'):
        """
        Initialize the CARLA route plotter.
        
        Args:
            host: CARLA server host
            port: CARLA server port
            town: CARLA town to load
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)
        self.world = self.client.load_world(town)
        self._set_sync_mode(True)
        
        # Initialize map and planner
        self.map = self.world.get_map()
        self.planner = GlobalRoutePlanner(self.map, resolution=1.0)
        
        # Default camera settings
        self.image_width = 512
        self.image_height = 512
        self.camera_fov = 90
        self.camera_x = 197.14505004882812
        self.camera_y = 164.2880915403366
        self.traj_radius = 199.2049560546875
        self.camera_z = 1.2 * np.tan((90 - self.camera_fov / 2) * np.pi / 180) * self.traj_radius
        
        # Initialize base image
        self.base_image = None
        self.current_image = None
        
    def _set_sync_mode(self, sync: bool) -> None:
        """Set synchronous mode for the CARLA world."""
        settings = self.world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 1.0 / 10.0
        self.world.apply_settings(settings)
    
    def _create_camera(self) -> 'Camera':
        """Create a camera sensor in the CARLA world."""
        return Camera(
            self.world,
            self.image_width,
            self.image_height,
            self.camera_fov,
            self.camera_x,
            self.camera_y,
            self.camera_z,
            -90,
            0
        )
    
    def _create_base_image(self) -> None:
        """Create the initial base image from CARLA."""
        camera = self._create_camera()
        self.world.tick()
        result = camera.get()
        self.base_image = Image.fromarray(result)
        self.current_image = self.base_image.copy()
    
    def _plot_route(self, 
                    global_plan_world_coord: List,
                    route_color: Optional[Tuple[int, int, int]] = None) -> None:
        """Plot a single route on the current image."""
        if route_color is None:
            route_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        
        draw = ImageDraw.Draw(self.current_image)
        
        meters_to_pixel_x = self.image_width / (2 * np.tan(self.camera_fov * np.pi / 180 / 2) * self.camera_z)
        meters_to_pixel_y = -self.image_height / (2 * np.tan(self.camera_fov * np.pi / 180 / 2) * self.camera_z)
        
        # Plot first point
        # last_point = global_plan_world_coord[0][0].transform.location
        last_point = global_plan_world_coord[0][0].location
        last_point_x = meters_to_pixel_x * (last_point.y - self.camera_y) + self.image_width / 2
        last_point_y = meters_to_pixel_y * (last_point.x - self.camera_x) + self.image_height / 2
        
        radius = 2
        draw.ellipse([last_point_x - radius, last_point_y - radius,
                      last_point_x + radius, last_point_y + radius],
                     fill=route_color)
        
        # Plot route
        for point_idx in range(1, len(global_plan_world_coord)):
            # point_loc = global_plan_world_coord[point_idx][0].transform.location
            point_loc = global_plan_world_coord[point_idx][0].location
            point_x = meters_to_pixel_x * (point_loc.y - self.camera_y) + self.image_width / 2
            point_y = meters_to_pixel_y * (point_loc.x - self.camera_x) + self.image_height / 2
            draw.line((last_point_x, last_point_y, point_x, point_y), width=2, fill=route_color)
            last_point_x = point_x
            last_point_y = point_y
        
        # Plot last point
        draw.ellipse([last_point_x - radius, last_point_y - radius,
                      last_point_x + radius, last_point_y + radius],
                     fill=route_color)
    
    # def _process_route(self, route_coords: List[Tuple[float, float, float]]) -> List:
    #     """Process route coordinates into a CARLA route plan."""
    #     transforms = []
    #     for x, y, z in route_coords:
    #         location = carla.Location(x=x, y=y, z=z)
    #         transform = carla.Transform(location, carla.Rotation())
    #         transforms.append(transform)
        
    #     spawn_transform = transforms[0]
    #     target_transforms = transforms[1:]
    #     current_location = spawn_transform.location
    #     global_plan_world_coord = []
        
    #     for tt in target_transforms:
    #         next_target_location = tt.location
    #         route_trace = self.planner.trace_route(current_location, next_target_location)
    #         global_plan_world_coord += route_trace
    #         current_location = next_target_location
            
    #     return global_plan_world_coord
    
    def _process_route(self, route_coords: List[Tuple[float, float, float]]) -> List:
        """
        Process route coordinates into a CARLA route plan.
        For dense trajectory data, preserves the original path without interpolation.
        """
        global_plan_world_coord = []
        
        # Convert each coordinate to a CARLA transform
        for x, y, z in route_coords:
            location = carla.Location(x=x, y=y, z=z)
            transform = carla.Transform(location, carla.Rotation())
            # Create a dummy waypoint for compatibility with the existing plotting code
            # The RoadOption.LANEFOLLOW is just a placeholder
            global_plan_world_coord.append((transform, RoadOption.LANEFOLLOW))
        
        return global_plan_world_coord
    
    def plot_routes_from_file(self, file_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Plot routes from a text file containing route coordinates.
        
        Args:
            file_path: Path to the text file containing route coordinates
            output_dir: Directory to save the output images
        """
        # Load routes
        with open(file_path, 'r') as f:
            routes = ast.literal_eval(f.read())
        
        self.plot_routes_from_list(routes, output_dir)

    def plot_routes_from_file_eval(self, file_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Plot routes from a text file containing route coordinates.

        Args:
            file_path: Path to the text file containing route coordinates
            output_dir: Directory to save the output images
        """
        routes = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    x, y, z = map(float, parts)
                    routes.append([x, y, z])

        self.plot_routes_from_list([routes], output_dir)


    def plot_routes_from_list(self, 
                            routes: List[List[Tuple[float, float, float]]], 
                            path: Union[str, Path]) -> None:
        """
        Plot routes from a list of route coordinates.
        
        Args:
            routes: List of routes, where each route is a list of (x, y, z) coordinates
            output_dir: Directory to save the output images
        """
        # output_dir = Path(output_dir)
        # output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create base image if not exists
        if self.base_image is None:
            self._create_base_image()
        
        # Reset current image to base image
        self.current_image = self.base_image.copy()
        
        # Plot each route
        for route_id, route_coords in enumerate(routes):
            global_plan_world_coord = self._process_route(route_coords)
            self._plot_route(global_plan_world_coord)
            
            # Save current state
            # image_path = path / f'routes_0_to_{route_id:02d}.png'
            image_path = path
            # self.current_image.save(image_path.as_posix())
            self.current_image.save(image_path)

    def load_multiple_routes(self, file_paths: List[Union[str, Path]], output_dir: Union[str, Path]) -> List[List[List[float]]]:
        """
        Carrega múltiplas rotas a partir de arquivos .txt, cada um contendo uma rota.

        Args:
            file_paths: Lista de caminhos para os arquivos .txt

        Returns:
            Uma lista de rotas, onde cada rota é uma lista de coordenadas [x, y, z]
        """
        all_routes = []

        for file_path in file_paths:
            route = []
            if 'distance_traveled' in file_path:
                continue
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        x, y, z = map(float, parts)
                        route.append([x, y, z])
            all_routes.append(route)
        
        self.plot_routes_from_list(all_routes, output_dir)

        return all_routes


    def set_camera_params(self,
                         image_width: int = None,
                         image_height: int = None,
                         camera_fov: float = None,
                         camera_x: float = None,
                         camera_y: float = None,
                         camera_z: float = None) -> None:
        """
        Update camera parameters. Only updates provided parameters.
        
        Args:
            image_width: Width of the output image
            image_height: Height of the output image
            camera_fov: Field of view of the camera
            camera_x: X coordinate of the camera
            camera_y: Y coordinate of the camera
            camera_z: Z coordinate of the camera
        """
        params_updated = False
        if image_width is not None:
            self.image_width = image_width
            params_updated = True
        if image_height is not None:
            self.image_height = image_height
            params_updated = True
        if camera_fov is not None:
            self.camera_fov = camera_fov
            params_updated = True
        if camera_x is not None:
            self.camera_x = camera_x
            params_updated = True
        if camera_y is not None:
            self.camera_y = camera_y
            params_updated = True
        if camera_z is not None:
            self.camera_z = camera_z
            params_updated = True
        
        # If any parameters were updated, reset the base image
        if params_updated:
            self.base_image = None
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'world'):
            self._set_sync_mode(False)


class Camera:
    """Camera sensor class (unchanged from original implementation)"""
    def __init__(self, world, w, h, fov, x, y, z, pitch, yaw):
        bp_library = world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(w))
        camera_bp.set_attribute('image_size_y', str(h))
        camera_bp.set_attribute('fov', str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.queue = queue.Queue()
        self.camera = world.spawn_actor(camera_bp, transform)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None
        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def __del__(self):
        if hasattr(self, 'camera'):
            self.camera.destroy()


import os
import re
from typing import List, Union
from pathlib import Path

def get_txt_files_with_number(path: Union[str, Path]) -> List[str]:
    """
    Retorna todos os arquivos .txt de um diretório cujo nome termina com um número antes de '.txt'.

    Args:
        path: Caminho do diretório onde procurar os arquivos

    Returns:
        Lista de caminhos completos dos arquivos que satisfazem a condição
    """
    path = Path(path)
    txt_files = []

    # Expressão regular para pegar arquivos que terminam com número antes do .txt
    pattern = re.compile(r".*?(\d+)\.txt$")

    for file in path.glob("*.txt"):
        if pattern.match(file.name):
            txt_files.append(str(file.resolve()))

    return txt_files

import os
import re
from typing import List, Union
from pathlib import Path

def get_txt_files_with_epoch(path: Union[str, Path], epoch: int) -> List[str]:
    """
    Retorna todos os arquivos .txt de um diretório cujo nome contém _ep_{epoch}_ antes da extensão.

    Args:
        path: Caminho do diretório onde procurar os arquivos
        epoch: Número da epoch a ser filtrado

    Returns:
        Lista de caminhos completos dos arquivos que satisfazem a condição
    """
    path = Path(path)
    txt_files = []

    # Regex que procura arquivos com _ep_<epoch>_ antes de .txt
    pattern = re.compile(rf".*_ep_{epoch}_.+\.txt$")

    for file in path.glob("*.txt"):
        if pattern.match(file.name):
            txt_files.append(str(file.resolve()))

    return txt_files

import pandas as pd

import pandas as pd
import os

def classify_last_y_values(file_paths, output_csv_path):
    """
    Classify the direction based on the last Y-value from a list of .txt files.

    Args:
        file_paths (List[str]): List of paths to .txt files.
        output_csv_path (str): Path to save the output .csv file.
    """
    data = []

    for path in file_paths:
        try:
            with open(path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                
                if not lines:
                    continue
                
                last_line = lines[-1]
                parts = last_line.split()
                
                if len(parts) != 3:
                    continue
                
                y_value = float(parts[1])  # coluna do meio

                if y_value < 193:
                    direction = "LEFT"
                elif y_value > 199:
                    direction = "RIGHT"
                else:
                    direction = "STRAIGHT"

                filename = os.path.basename(path)
                data.append({
                    "filename": filename,
                    "direction": direction,
                    "value": y_value
                })
        
        except Exception as e:
            print(f"Erro ao processar {path}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV salvo em: {output_csv_path}")



def plot_direction_histogram_from_csvs(folder_path, output_image_path, min_epoch=20, max_epoch=600):
    """
    Plota histogramas empilhados normalizados (porcentagem) de direções (LEFT, STRAIGHT, RIGHT) para arquivos .csv.
    
    Args:
        folder_path (str): Caminho da pasta onde estão os arquivos CARLA_{EPOCH}.csv.
        output_image_path (str): Caminho para salvar a imagem do gráfico gerado.
        min_epoch (int): Epoch mínima a considerar (inclusive).
        max_epoch (int): Epoch máxima a considerar (inclusive).
    """
    epoch_files = []
    pattern = re.compile(r"CARLA_(\d+)\.csv")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if min_epoch <= epoch <= max_epoch:
                epoch_files.append((epoch, os.path.join(folder_path, filename)))

    # Ordenar por epoch
    epoch_files.sort()

    directions_count = {
        "LEFT": [],
        "STRAIGHT": [],
        "RIGHT": []
    }

    x_labels = []

    for epoch, filepath in epoch_files:
        df = pd.read_csv(filepath)
        total = len(df)
        if total == 0:
            continue

        left = (df["direction"] == "LEFT").sum()
        straight = (df["direction"] == "STRAIGHT").sum()
        right = (df["direction"] == "RIGHT").sum()

        directions_count["LEFT"].append(left / total)
        directions_count["STRAIGHT"].append(straight / total)
        directions_count["RIGHT"].append(right / total)
        x_labels.append(epoch)

    # Dados empilhados para plot
    lefts = np.array(directions_count["LEFT"])
    straights = np.array(directions_count["STRAIGHT"])
    rights = np.array(directions_count["RIGHT"])

    x = np.arange(len(x_labels))

    # Plotando o gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, lefts, label="LEFT", color="tab:blue")
    ax.bar(x, straights, bottom=lefts, label="STRAIGHT", color="tab:orange")
    ax.bar(x, rights, bottom=lefts + straights, label="RIGHT", color="tab:green")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylim(0, 1)
    # ax.set_title("normalized to percentage")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Epoch")
    ax.legend()
    plt.tight_layout()

    plt.savefig(output_image_path)
    plt.close()
    print(f"Gráfico salvo em: {output_image_path}")



if __name__ == '__main__':

    for round in [1,2,3,4,5]:
        caminho = f"diff_bc_video_(diffuser)/birdview/t_interction/BC/BC_Multi_Simple_0{round}_0_extra_steps/BC_Multi_Simple_0{round}/"
        plot_direction_histogram_from_csvs(caminho, caminho + f'CARLA_plot_histogram.png')
        caminho = caminho = f"diff_bc_video_(diffuser)/birdview/t_interction/Diffusion-BC/Diffusion_BC_Multi_Simple_0{round}_0_extra_steps/Diffusion_BC_Multi_Simple_0{round}/"
        plot_direction_histogram_from_csvs(caminho, caminho + f'CARLA_plot_histogram.png')

    plotter = CarlaRoutePlotter(host='localhost', port=2020, town='Town01')

    # for round in [1,2,3,4,5]:
    #     for ep in [20, 40, 80, 150, 250, 500, 600, 749]:
    #         caminho = f"diff_bc_video_(diffuser)/birdview/t_interction/Diffusion-BC/Diffusion_BC_Multi_Simple_0{round}_0_extra_steps/Diffusion_BC_Multi_Simple_0{round}/"
    #         epoch = ep

    #         lista_de_txts = get_txt_files_with_epoch(caminho, epoch)
    #         classify_last_y_values(lista_de_txts, caminho + f'CARLA_{ep}.csv')
    #         # Initialize the plotter

    #         plotter.load_multiple_routes(lista_de_txts,
    #                                     caminho + f'CARLA_{ep}.png')

    # Option 1: Plot routes from a file
    plotter.plot_routes_from_file_eval('diff_bc_video_(diffuser)/birdview/t_interction/BC/BC_Multi_Simple_01_0_extra_steps/BC_Multi_Simple_01/Model_cnn_BC_gail_experts_multi_bruno_3_simples_birdviewt_BC_067e_ep_1_0.txt', 'fig.png')


    for round in [1,2,3,4,5]:
        for ep in [1, 20, 40, 80, 150, 250, 500, 600, 749]:
            caminho = f"diff_bc_video_(diffuser)/birdview/t_interction/BC/BC_Multi_Simple_0{round}_0_extra_steps/BC_Multi_Simple_0{round}/"
            epoch = ep

            lista_de_txts = get_txt_files_with_epoch(caminho, epoch)
            classify_last_y_values(lista_de_txts, caminho + f'CARLA_{ep}.csv')
            # Initialize the plotter

            plotter.load_multiple_routes(lista_de_txts,
                                        caminho + f'CARLA_{ep}.png')

    # Option 1: Plot routes from a file
    plotter.plot_routes_from_file_eval('diff_bc_video_(diffuser)/birdview/t_interction/BC/BC_Multi_Simple_01_0_extra_steps/BC_Multi_Simple_01/Model_cnn_BC_gail_experts_multi_bruno_3_simples_birdviewt_BC_067e_ep_1_0.txt', 'fig.png')