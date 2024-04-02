#!/usr/bin/env python

# Copyright (c) 2022 authors.
# authors: Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu and Luc van Gool
#
# This work is licensed under the terms of the CC-BY-NC 4.0.
# For a copy, see <https://creativecommons.org/licenses/by-nc/4.0/deed>.

from carla_gym import CARLA_GYM_ROOT_DIR
from carla_gym.carla_multi_agent_env import CarlaMultiAgentEnv
from carla_gym.utils import config_utils
import json


class LeaderboardEnv(CarlaMultiAgentEnv):
    def __init__(self, carla_map, host, port, seed, no_rendering, obs_configs, reward_configs, terminal_configs,
                 weather_group, routes_group, routes_file=None):

        all_tasks = self.build_all_tasks(carla_map, weather_group, routes_group, routes_file)
        super().__init__(carla_map, host, port, seed, no_rendering,
                         obs_configs, reward_configs, terminal_configs, all_tasks)

    @staticmethod
    def build_all_tasks(carla_map, weather_group, routes_group, routes_file):
        assert carla_map in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']
        num_zombie_vehicles = {
            'Town01': 120,
            'Town02': 70,
            'Town03': 70,
            'Town04': 150,
            'Town05': 120,
            'Town06': 120
        }
        num_zombie_walkers = {
            'Town01': 120,
            'Town02': 70,
            'Town03': 70,
            'Town04': 80,
            'Town05': 120,
            'Town06': 80
        }

        # weather
        if weather_group == 'new':
            weathers = ['SoftRainSunset', 'WetSunset']
        elif weather_group == 'train':
            weathers = ['ClearNoon', 'WetNoon', 'HardRainNoon', 'ClearSunset']
        elif weather_group == 'simple':
            weathers = ['ClearNoon']
        elif weather_group == 'train_eval':
            weathers = ['WetNoon', 'ClearSunset']
        elif weather_group == 'all':
            weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon',
                        'HardRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset',
                        'SoftRainSunset', 'MidRainSunset', 'HardRainSunset']
        else:
            weathers = [weather_group]

        # task_type setup
        if carla_map == 'Town04' and routes_group is not None:
            description_folder = CARLA_GYM_ROOT_DIR / 'envs/scenario_descriptions/LeaderBoard' \
                / f'Town04_{routes_group}'
        elif carla_map == 'Town01' and routes_group is 'multi':
            description_folder = CARLA_GYM_ROOT_DIR / 'envs/scenario_descriptions/LeaderBoard' \
                / f'Town01_{routes_group}'
        else:
            description_folder = CARLA_GYM_ROOT_DIR / 'envs/scenario_descriptions/LeaderBoard' / carla_map

        print(description_folder)
        actor_configs_dict = json.load(open(description_folder / 'actors.json'))
        if routes_file is None:
            route_descriptions_dict = config_utils.parse_routes_file(description_folder / 'routes.xml')
        else:
            route_descriptions_dict = config_utils.parse_routes_file(description_folder / routes_file)

        all_tasks = []
        print(len(route_descriptions_dict))
        for weather in weathers:
            for route_id, route_description in route_descriptions_dict.items():
                task = {
                    'weather': weather,
                    'description_folder': description_folder,
                    'route_id': route_id,
                    'num_zombie_vehicles': num_zombie_vehicles[carla_map],
                    'num_zombie_walkers': num_zombie_walkers[carla_map],
                    'ego_vehicles': {
                        'routes': route_description['ego_vehicles'],
                        'actors': actor_configs_dict['ego_vehicles'],
                    },
                    'scenario_actors': {
                        'routes': route_description['scenario_actors'],
                        'actors': actor_configs_dict['scenario_actors']
                    } if 'scenario_actors' in actor_configs_dict else {}
                }
                all_tasks.append(task)

        return all_tasks
