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


class Camera(object):
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
        pass


def process_img(image):
    image.convert(cc.Raw)
    image.save_to_disk('_out/%08d' % image.frame_number)


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


def load_routes_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    routes = ast.literal_eval(content)
    
    route_descriptions_dict = {}
    for route_id, route_coords in enumerate(routes):
        transforms = []
        for x, y, z in route_coords:
            location = carla.Location(x=x, y=y, z=z)
            transform = carla.Transform(location, carla.Rotation())
            transforms.append(transform)
        
        route_descriptions_dict[route_id] = {
            'ego_vehicles': {
                'hero': transforms
            }
        }
    
    return route_descriptions_dict


def create_base_image(world, camera_x, camera_y, camera_z, image_width, image_height, camera_fov):
    """Creates the initial base image from CARLA"""
    camera = Camera(world, image_width, image_height, camera_fov, camera_x, camera_y, camera_z, -90, 0)
    world.tick()
    result = camera.get()
    return Image.fromarray(result)

def plot_route(draw, global_plan_world_coord, camera_x, camera_y, camera_z, image_width, image_height, camera_fov, route_color=None):
    """Plots a single route on the given ImageDraw object"""
    if route_color is None:
        route_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    
    meters_to_pixel_x = image_width / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
    meters_to_pixel_y = -image_height / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
    
    # Plot first point
    last_point = global_plan_world_coord[0][0].transform.location
    last_point_x = meters_to_pixel_x * (last_point.y - camera_y) + image_width / 2
    last_point_y = meters_to_pixel_y * (last_point.x - camera_x) + image_height / 2
    
    radius = 2
    draw.ellipse([last_point_x - radius, last_point_y - radius, 
                  last_point_x + radius, last_point_y + radius], 
                 fill=route_color)

    # Plot route
    for point_idx in range(1, len(global_plan_world_coord)):
        point_loc = global_plan_world_coord[point_idx][0].transform.location
        point_x = meters_to_pixel_x * (point_loc.y - camera_y) + image_width / 2
        point_y = meters_to_pixel_y * (point_loc.x - camera_x) + image_height / 2
        draw.line((last_point_x, last_point_y, point_x, point_y), width=2, fill=route_color)
        last_point_x = point_x
        last_point_y = point_y
    
    # Plot last point
    draw.ellipse([last_point_x - radius, last_point_y - radius, 
                  last_point_x + radius, last_point_y + radius], 
                 fill=route_color)

def main():
    # Setup
    host = 'localhost'
    port = 2020
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.load_world('Town01')
    set_sync_mode(client, True)
    
    # Initialize parameters
    image_width = 512
    image_height = 512
    camera_fov = 90
    camera_x = 197.14505004882812
    camera_y = 164.2880915403366
    traj_radius = 199.2049560546875
    camera_z = 1.2 * np.tan((90 - camera_fov / 2) * np.pi / 180) * traj_radius
    
    # Setup directories
    traj_output_dir = Path('paper_plots/traj_plot')
    traj_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load routes
    routes_file = Path('town01_routes.txt')
    route_descriptions_dict = load_routes_from_file(routes_file)
    
    # Create base image
    base_image = create_base_image(world, camera_x, camera_y, camera_z, 
                                 image_width, image_height, camera_fov)
    current_image = base_image.copy()
    
    # Setup route planner
    map = world.get_map()
    planner = GlobalRoutePlanner(map, resolution=1.0)
    
    # Iteratively plot and save each route
    for route_id, route_description in route_descriptions_dict.items():
        # Get route coordinates
        route_config = route_description['ego_vehicles']
        spawn_transform = route_config['hero'][0]
        target_transforms = route_config['hero'][1:]
        current_location = spawn_transform.location
        
        # Calculate route
        global_plan_world_coord = []
        for tt in target_transforms:
            next_target_location = tt.location
            route_trace = planner.trace_route(current_location, next_target_location)
            global_plan_world_coord += route_trace
            current_location = next_target_location
        ds_ids = downsample_route(global_plan_world_coord, 50)
        
        # Plot route on current image
        draw = ImageDraw.Draw(current_image)
        plot_route(draw, global_plan_world_coord, camera_x, camera_y, camera_z,
                  image_width, image_height, camera_fov)
        
        # Save current state
        image_path = traj_output_dir / f'routes_0_to_{route_id:02d}.png'
        current_image.save(image_path.as_posix())

if __name__ == '__main__':
    main()