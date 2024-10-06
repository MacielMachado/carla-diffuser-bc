# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/utils/rl_birdview_wrapper.py

import gym
import numpy as np
import cv2
import carla
import pandas as pd
from pathlib import Path
from PIL import Image

import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util


eval_num_zombie_vehicles = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 150,
    'Town05': 120,
    'Town06': 120
}
eval_num_zombie_walkers = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 80,
    'Town05': 120,
    'Town06': 80
}

class RlBirdviewWrapper(gym.Wrapper):
    def __init__(self, env):
        self._ev_id = list(env._obs_configs.keys())[0]  # Hero mode gives birdview, central_rgb, control, gnss, left_rgb, right_rgb, route_plan, speed, velocity
        self._render_dict = {}

        observation_space = {}
        observation_space['birdview'] = env.observation_space[self._ev_id]['birdview']['masks']
        observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)  # We create this new observation dimension
        observation_space['left_rgb'] = env.observation_space[self._ev_id]['left_rgb']['data']
        observation_space['right_rgb'] = env.observation_space[self._ev_id]['right_rgb']['data']
        observation_space['central_rgb'] = env.observation_space[self._ev_id]['central_rgb']['data']
        observation_space['gnss'] = env.observation_space['hero']['gnss']['gnss']
        observation_space['speed'] = env.observation_space['hero']['speed']
        observation_space['velocity'] = env.observation_space['hero']['velocity']

        # observation_space['birdview'] = env.observation_space['birdview']
        # observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)  # We create this new observation dimension
        # observation_space['left_rgb'] = env.observation_space['left_rgb']
        # observation_space['right_rgb'] = env.observation_space['right_rgb']
        # observation_space['central_rgb'] = env.observation_space['central_rgb']
        # observation_space['gnss'] = env.observation_space['gnss']

        env.observation_space = gym.spaces.Dict(**observation_space)

        env.action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)  # Define action dimensionality

        super(RlBirdviewWrapper, self).__init__(env)

        self.eval_mode = True
        self.env_world = env

    def get_world(self):
        return self.env_world._world

    def reset(self, observation_type='birdview'):
        if self.eval_mode:
            self.env.eval_mode = True
            self.env._task['num_zombie_vehicles'] = eval_num_zombie_vehicles[self.env._carla_map]
            self.env._task['num_zombie_walkers'] = eval_num_zombie_walkers[self.env._carla_map]
            for ev_id in self.env._ev_handler._terminal_configs:
                self.env._ev_handler._terminal_configs[ev_id]['kwargs']['eval_mode'] = True
        else:
            self.env.eval_mode = False
            for ev_id in self.env._ev_handler._terminal_configs:
                self.env._ev_handler._terminal_configs[ev_id]['kwargs']['eval_mode'] = False

        obs_ma = self.env.reset()
        action_ma = {self._ev_id: carla.VehicleControl(manual_gear_shift=True, gear=1)}
        obs_ma, _, _, _ = self.env.step(action_ma)
        action_ma = {self._ev_id: carla.VehicleControl(manual_gear_shift=False)}
        obs_ma, _, _, _ = self.env.step(action_ma)

        snap_shot = self.env._world.get_snapshot()
        self.env._timestamp = {
            'step': 0,
            'frame': 0,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        obs = self.process_obs(obs_ma[self._ev_id], observation_type=observation_type)

        self._render_dict['prev_obs'] = obs
        self._render_dict['prev_im_render'] = obs_ma[self._ev_id]['birdview']['rendered']
        return obs

    def step(self, action):
        action_ma = {self._ev_id: self.process_act(action)}

        obs_ma, reward_ma, done_ma, info_ma = self.env.step(action_ma)

        obs = self.process_obs(obs_ma[self._ev_id])

        reward = reward_ma[self._ev_id]
        done = done_ma[self._ev_id]
        info = info_ma[self._ev_id]

        self._render_dict = {
            'timestamp': self.env.timestamp,
            'obs': self._render_dict['prev_obs'],
            'prev_obs': obs,
            'im_render': self._render_dict['prev_im_render'],
            'prev_im_render': obs_ma[self._ev_id]['birdview']['rendered'],
            'action': action,
            'reward_debug': info['reward_debug'],
            'terminal_debug': info['terminal_debug']
        }
        info['left_rgb'] = obs['left_rgb']
        info['central_rgb'] = obs['central_rgb']
        info['right_rgb'] = obs['right_rgb']
        info['gnss'] = obs['gnss']

        info['collisions_layout'] = info_ma['collisions_layout']
        info['collisions_vehicle'] = info_ma['collisions_vehicle']
        info['collisions_pedestrian'] = info_ma['collisions_pedestrian']
        info['collisions_others'] = info_ma['collisions_others']
        info['route_deviation'] = info_ma['route_deviation']
        info['wrong_lane'] = info_ma['wrong_lane']
        info['outside_lane'] = info_ma['outside_lane']
        info['run_red_light'] = info_ma['run_red_light']
        info['encounter_stop'] = info_ma['encounter_stop']
        info['stop_infraction'] = info_ma['stop_infraction']
        return obs, reward, done, info
        # return obs, '', '', ''

    def render(self, mode='human', diff=True):
        '''
        train render: used in train_rl.py
        '''
        if diff:
            self.action_log_probs = [0.0]
            self.action_mu = [0.0]
            self.action_sigma = [0.0]
        self._render_dict['action_log_probs'] = self.action_log_probs
        self._render_dict['action_mu'] = self.action_mu
        self._render_dict['action_sigma'] = self.action_sigma
        return self.im_render(self._render_dict, diff=diff)

    @staticmethod
    def im_render(render_dict, diff=True):
        im_birdview = render_dict['im_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        if not diff:
            action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)
            mu_str = np.array2string(render_dict['action_mu'], precision=2, separator=',', suppress_small=True)
            sigma_str = np.array2string(render_dict['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(render_dict['obs']['state'], precision=2, separator=',', suppress_small=True)

        txt_t = f'step:{render_dict["timestamp"]["step"]:5}, frame:{render_dict["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (3, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        if not diff:
            txt_3 = f'a{mu_str} b{sigma_str}'
            im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for i, txt in enumerate(render_dict['reward_debug']['debug_texts'] +
                                render_dict['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im

    @staticmethod
    def process_obs(obs, observation_type='birdview'):
        obs_dict = {}
        state_list = []
        state_list.append(obs['control']['throttle'])
        state_list.append(obs['control']['steer'])
        state_list.append(obs['control']['brake'])
        state_list.append(obs['control']['gear']/5.0)
        state_list.append(obs['velocity']['vel_xy'])
        state_list.append(obs['speed']['speed'])
        state_list.append(obs['speed']['forward_speed'])
        obs_dict['state'] = np.concatenate(state_list)

        birdview = obs['birdview']['masks']
        obs_dict.update({
            'birdview': birdview
        })

        central_rgb = obs['central_rgb']['data']
        central_rgb = np.transpose(central_rgb, [2, 0, 1])

        left_rgb = obs['left_rgb']['data']
        left_rgb = np.transpose(left_rgb, [2, 0, 1])

        right_rgb = obs['right_rgb']['data']
        right_rgb = np.transpose(right_rgb, [2, 0, 1])

        gnss = obs['gnss']['gnss']

        speed = obs['speed']['speed'].item()
        forward_speed = obs['speed']['forward_speed'].item()

        obs_dict.update({
            'central_rgb': central_rgb,
            'left_rgb': left_rgb,
            'right_rgb': right_rgb,
            'gnss': gnss,
            'speed': speed,
            'forward_speed': forward_speed
        })

        return obs_dict

    @staticmethod
    def process_act(action):
        acc, steer = action.astype(np.float64)
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control
