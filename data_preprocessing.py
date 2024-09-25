import os
import cv2
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize

class DataHandler():
    def __init__(self):
        pass

    @staticmethod
    def load_data(path):
        return np.load(path).astype('float32')

    def append_data(self, array_1, array_2):
        return np.append(array_1, array_2, axis=0)

    def frac_array(self, array, frac):
        length = len(array)
        return np.array(array[:int((1-frac) * length)]),\
               np.array(array[int((1-frac) * length):])

    def to_greyscale(self, imgs):
        if imgs.shape[-1] == 6:
            birdview = np.dot(imgs[:,:,:,:3], [0.2989, 0.5870, 0.1140])
            semaphore = np.dot(imgs[:,:,:,3:], [0.2989, 0.5870, 0.1140])
            return np.stack((birdview, semaphore),axis=3)
        return np.dot(imgs, [0.2989, 0.5870, 0.1140])
        
    def normalizing(self, imgs):
        return imgs/255.0
    
    def plot_batch(self, observations, actions, batch_size, render=False):
        ''' Plot the states recordings.
        '''
        if not(render): return
        fig, ax = plt.subplots(8,4)
        for i in range(batch_size):
            col = math.floor(i/4)
            row = i - 4*col
            frame = i
            ax[col, row].imshow(observations[frame], cmap=plt.get_cmap("gray"))
            ax[col, row].title.set_text(f'Frame: {i}, Actions: {list(actions[frame].numpy())}')
            ax[col, row].axis('off')
        plt.show()

    def green_mask(self, observation):
        
        #convert to hsv
        hsv = cv2.cvtColor(observation, x)
        
        mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

        #slice the green
        imask_green = mask_green>0
        green = np.zeros_like(observation, np.uint8)
        green[imask_green] = observation[imask_green]
        
        return(green)
    
    def gray_scale(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return gray
    
    def blur_image(self, observation):
        blur = cv2.GaussianBlur(observation, (5, 5), 0)
        return blur

    def canny_edge_detector(self, observation):
        canny = cv2.Canny(observation, 50, 150)
        return canny

    def stack_with_previous(self, images_array, num_images=4):
        if len(images_array.shape) == 3:
            images_array = np.expand_dims(images_array, axis=-1)
        batch_size, height, width, channels = images_array.shape
        stacked_images = np.zeros((batch_size, height, width, channels * num_images), dtype=images_array.dtype)

        for i in range(batch_size):
            if i < num_images-1:
                    stacked_images[i, :, :, :] = np.concatenate([images_array[i] for _ in range(num_images)], axis=-1)
            else:
                stacked_images[i, :, :, :] = np.concatenate([images_array[i - j] for j in range(num_images)], axis=-1)

        return stacked_images
    
    def stack_with_previous_front(self, images_array, seq):
        pass

    def get_images_array(self, images_array, observation_type):
        new_list = np.array([])
        for i in tqdm(range(len(images_array)), desc=f'Analyzing {observation_type} data'):
        # for i in tqdm(range(5), desc=f'Analyzing {observation_type} data'):
            # new_list.append(images_array[i][0][observation_type])
            if i == 0:
                new_list = np.expand_dims(np.array(images_array[i][0][observation_type]), axis=0)
            else:
                arr = np.expand_dims(np.array(images_array[i][0][observation_type]), axis=0)
                new_list = np.concatenate((new_list, arr), axis=0)
            # if i > 5:
            #     return np.array(new_list)

        return np.array(new_list)

    def __preprocess_front_images(self, images_array, eval, stack_with_previous=True):
        # obs_front = images_array['central_rgb'] if eval else np.array([np.array(ele[0]['central_rgb']) for ele in images_array])
        obs_front = np.expand_dims(images_array['central_rgb'], 0) if eval else self.get_images_array(images_array, 'central_rgb')
        obs_front = np.transpose(obs_front, (0, 2, 3, 1))
        obs_front = DataHandler().to_greyscale(obs_front)
        obs_front = DataHandler().normalizing(obs_front)

        # obs_left = images_array['left_rgb'] if eval else np.array([np.array(ele[0]['left_rgb']) for ele in images_array])
        obs_left = np.expand_dims(images_array['left_rgb'], 0) if eval else self.get_images_array(images_array, 'left_rgb')
        obs_left = np.transpose(obs_left, (0, 2, 3, 1))
        obs_left = DataHandler().to_greyscale(obs_left)
        obs_left = DataHandler().normalizing(obs_left)

        # obs_right = images_array['right_rgb'] if eval else np.array([np.array(ele[0]['right_rgb']) for ele in images_array])
        obs_right = np.expand_dims(images_array['right_rgb'], 0) if eval else self.get_images_array(images_array, 'right_rgb')
        obs_right = np.transpose(obs_right, (0, 2, 3, 1))
        obs_right = DataHandler().to_greyscale(obs_right)
        obs_right = DataHandler().normalizing(obs_right)
        
        if stack_with_previous:
            obs_front = DataHandler().stack_with_previous(obs_front, num_images=4)
            obs_left = DataHandler().stack_with_previous(obs_left, num_images=4)
            obs_right = DataHandler().stack_with_previous(obs_right, num_images=4)
        else:
            obs_front = np.expand_dims(obs_front, -1)
            obs_left = np.expand_dims(obs_left, -1)
            obs_right = np.expand_dims(obs_right, -1)

        obs_stack = np.concatenate((obs_left, obs_front, obs_right), axis=3)
        obs_stack_resized = []
        for i in tqdm(range(len(obs_stack)), desc='Image Resize'):
            obs_stack_resized.append(resize(
                obs_stack[i,:,:,:],
                (224, 224, obs_stack.shape[-1]),
                mode='constant'))
        return np.array(obs_stack_resized)

    def __resize(self, obs):
        obs_resized = []
        for i in tqdm(range(len(obs)), desc='Image Resize'):
            obs_resized.append(resize(
                obs[i,:,:,:],
                (224, 224, obs.shape[-1]),
                mode='constant'))
        return np.array(obs_resized)

    def __preprocess_birdview(self, images_array, eval=False, embedding='Model_cnn_mlp'):
        obs = np.expand_dims(images_array['birdview'], 0) if eval else np.array([np.array(ele[0]['birdview']) for ele in images_array])
        obs = np.transpose(obs, (0, 2, 3, 1))
        obs = DataHandler().to_greyscale(obs)
        obs = DataHandler().normalizing(obs)
        stack_size = 3 if embedding == 'Model_cnn_mlp_resnet18' else 4
        obs = DataHandler().stack_with_previous(obs, stack_size)
        return self.__resize(obs) if embedding == 'Model_cnn_mlp_resnet18' else obs
    
    def __preprocess_human_images(self, images_array):
        images = DataHandler().to_greyscale(images_array)
        images = DataHandler().normalizing(images)
        images = DataHandler().stack_with_previous(images)
        return images
    
    def preprocess_images(self, images_array, observation_type: str, eval=False, embedding='Model_cnn_mlp', stack_with_previous=True):
        if observation_type == 'birdview':
            return self.__preprocess_birdview(images_array, eval, embedding)
        elif observation_type == 'human':
            return self.__preprocess_human_images(images_array)
        elif observation_type == 'front':
            return self.__preprocess_front_images(images_array, eval, stack_with_previous)
        raise NotImplementedError
    
    def preprocess_actions(self, actions_array, origin):
        if origin == 'ppo':
            return  self.preprocess_ppo_actions(actions_array)
        if origin == 'human':
            return actions_array
        raise NotImplementedError
        
    def preprocess_ppo_actions(self, actions_array):
        steering_dim = actions_array[:,0]
        gas_dim = [val if val > 0 else 0 for val in actions_array[:,1]]
        brake_dim = [val if val < 0 else 0 for val in actions_array[:,1]]
        return np.array([steering_dim, gas_dim, brake_dim]).T


class CarlaCustomDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions):
        self.obs = obs
        self.actions = actions

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return torch.from_numpy(self.obs[index]), torch.from_numpy(self.actions[index])


class CarlaCustomDatasetSpeed(torch.utils.data.Dataset):
    def __init__(self, obs, actions, speed):
        self.obs = obs
        self.actions = actions
        self.speed = speed

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return torch.from_numpy(self.obs[index]), torch.from_numpy(self.actions[index]), torch.from_numpy(self.speed[index])


class DatasetHandler():
    def __init__(self):
        pass

    @classmethod
    def subdivide_array(self, array, division_indices, highest_values):
        # Inicialize uma lista vazia para armazenar os subitens
        subitems = np.array([])
        division_indices = [0] + division_indices
        # Itere sobre os índices em B para criar os subitens
        for i in range(0, len(division_indices) - 1):
            start_index = division_indices[i]
            end_index = division_indices[i+1]
            subitem = array[start_index:end_index]
            if division_indices[i] in highest_values:
                if len(subitems) == 0:
                    subitems = subitem
                    continue
                subitems = np.append(subitems, subitem, axis=0)
        return subitems
    
    @classmethod
    def highest_scores(self, rewards, num_ele):
        indices = np.argpartition(rewards[:,1], -num_ele)[-num_ele:]
        highest_rewards = rewards[indices]
        return highest_rewards
    

class BirdViewMovieMaker():
    def __init__(self, path):
        self.path = path

    def save_record(self):
        # Define video properties
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video format (check FourCC codes online)
        fps = 24  # Frames per second
        video_path = self.path + "video_output.mp4"  # Output video path

        # Get image size from the first image
        img_path = os.path.join(self.path, "0000_00.png")  # Assuming your images follow a naming convention
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        # Create video writer
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Loop through images and write to video
        for ele in sorted(os.listdir(self.path)):  # Assuming images are in "images" folder
            if ele[-6:-4] == '00':
                img_path = self.path + ele
                img = cv2.imread(img_path)
                video.write(img)

        # Release resources
        video.release()
        cv2.destroyAllWindows()

        print("Video created successfully!")


class FrontCameraMovieMakerArray():
    def __init__(self, video_path, front_array, left_array, right_array, birdview_array):
        self.video_path = video_path
        self.front_array = np.array(front_array)
        self.left_array = np.array(left_array)
        self.right_array = np.array(right_array)
        self.birdview_array = np.array(birdview_array)

    def save_record(self):
        # Define video properties
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video format (check FourCC codes online)
        fps = 24  # Frames per second
        video_path = self.video_path

        central_img = np.zeros((192, self.front_array[0].shape[1], self.front_array[0].shape[2]))
        central_img[:self.front_array.shape[1], :, :] = self.front_array[0]
        left_img = np.zeros((192, self.left_array[0].shape[1], self.left_array[0].shape[2]))
        left_img[:self.left_array.shape[1], :, :] = self.left_array[0]
        right_img = np.zeros((192, self.right_array[0].shape[1], self.right_array[0].shape[2]))
        right_img[:self.right_array.shape[1], :, :] = self.right_array[0]
        birdview_img = self.birdview_array[0]

        height, central_width, channels = central_img.shape

        _, left_width, _ = left_img.shape
        _, right_width, _ = right_img.shape
        _, birdview_width, _ = birdview_img.shape
        width = left_width + central_width + right_width + birdview_width
        img = np.hstack((left_img, central_img, right_img, birdview_img))

        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for i in range(len(self.front_array)):
            central_img_old = self.front_array[i]
            left_img_old = self.left_array[i]
            right_img_old = self.right_array
            birdview_img = self.birdview_array[i]

            # central_img = np.zeros((192, central_img_old.shape[1], central_img_old.shape[2]))
            # central_img[:central_img_old.shape[0], :, :] = central_img_old
            # left_img = np.zeros((192, left_img_old.shape[1], left_img_old.shape[2]))
            # left_img[:left_img_old.shape[0], :, :] = left_img_old
            # right_img = np.zeros((192, right_img_old.shape[1], right_img_old.shape[2]))
            # right_img[:right_img_old.shape[0], :, :] = right_img_old

            central_img = np.zeros((192, self.front_array[i].shape[1], self.front_array[i].shape[2]))
            central_img[:self.front_array.shape[1], :, :] = self.front_array[i]
            left_img = np.zeros((192, self.left_array[i].shape[1], self.left_array[i].shape[2]))
            left_img[:self.left_array.shape[1], :, :] = self.left_array[i]
            right_img = np.zeros((192, self.right_array[i].shape[1], self.right_array[i].shape[2]))
            right_img[:self.right_array.shape[1], :, :] = self.right_array[i]
            birdview_img = self.birdview_array[i]

            img = np.hstack((left_img, central_img, right_img, birdview_img))
            video.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

        video.release()
        cv2.destroyAllWindows()

        print("Video created successfully")

class FrontCameraMovieMaker():
    def __init__(self, path, name_index=''):
        self.path = path
        self.name_index = name_index

    def save_record(self):
        # Define video properties
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video format (check FourCC codes online)
        fps = 24  # Frames per second
        video_path = os.path.join(self.path.split("/")[0], self.path.split("/")[1],
                                  "video_output_dir_"+self.name_index+".mp4")  # Output video path

        # Get image size from the first image
        central_img_path = os.path.join(self.path, "central_rgb/0000.png")
        left_img_path = os.path.join(self.path, "left_rgb/0000.png")
        right_img_path = os.path.join(self.path, "right_rgb/0000.png")
        birdview_img_path = os.path.join(self.path, "birdview_masks/0000_00.png")

        central_img_old = cv2.imread(central_img_path)
        left_img_old = cv2.imread(left_img_path)
        right_img_old = cv2.imread(right_img_path)

        central_img = np.zeros((192, central_img_old.shape[1], central_img_old.shape[2]))
        central_img[:central_img_old.shape[0], :, :] = central_img_old
        left_img = np.zeros((192, left_img_old.shape[1], left_img_old.shape[2]))
        left_img[:left_img_old.shape[0], :, :] = left_img_old
        right_img = np.zeros((192, right_img_old.shape[1], right_img_old.shape[2]))
        right_img[:right_img_old.shape[0], :, :] = right_img_old

        birdview_img = cv2.imread(birdview_img_path)
        height, central_width, channels = central_img.shape

        _, left_width, _ = left_img.shape
        _, right_width, _ = right_img.shape
        _, birdview_width, _ = birdview_img.shape
        width = left_width + central_width + right_width + birdview_width
        img = np.hstack((left_img, central_img, right_img, birdview_img))

        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        front_path = os.path.join(self.path, "central_rgb")
        left_path = os.path.join(self.path, "left_rgb")
        right_path = os.path.join(self.path, "right_rgb")
        birdview_path = os.path.join(self.path, "birdview_masks")

        for ele in sorted(os.listdir(front_path)):
            central_img_old = cv2.imread(os.path.join(front_path, ele))
            left_img_old = cv2.imread(os.path.join(left_path, ele))
            right_img_old = cv2.imread(os.path.join(right_path, ele))
            birdview_img = cv2.imread(os.path.join(birdview_path, ele[:4]+'_00'+ele[-4:]))

            central_img = np.zeros((192, central_img_old.shape[1], central_img_old.shape[2]))
            central_img[:central_img_old.shape[0], :, :] = central_img_old
            left_img = np.zeros((192, left_img_old.shape[1], left_img_old.shape[2]))
            left_img[:left_img_old.shape[0], :, :] = left_img_old
            right_img = np.zeros((192, right_img_old.shape[1], right_img_old.shape[2]))
            right_img[:right_img_old.shape[0], :, :] = right_img_old

            img = np.hstack((left_img, central_img, right_img, birdview_img))
            video.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

        video.release()
        cv2.destroyAllWindows()

        print("Video created successfully")
 

class ActionsHistogram():
    def __init__(self):
        pass

    def main(self, path):
        actions = self.load_data(path)
        fig = self.gerar_histogramas(list(actions))
        self.save_fig(fig, path)

    @staticmethod
    def load_data(path):
        path = os.path.join(path, 'episode.json')
        return pd.read_json(path)['actions'].values
    
    def gerar_histogramas(self, actions):
        coluna_a = [sublista[0] for sublista in actions]
        coluna_b = [sublista[1] for sublista in actions]
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].hist(coluna_a, bins=10, color='blue', alpha=0.7)
        axs[0].set_title('Acceleration')
        # axs[0].set_xlabel('Valores de A')
        # axs[0].set_ylabel('Frequência')

        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([0, 500])

        axs[1].hist(coluna_b, bins=10, color='green', alpha=0.7)
        axs[1].set_title('Steering')
        # axs[1].set_xlabel('Valores de B')
        # axs[1].set_ylabel('Frequência')

        axs[1].set_xlim([-1, 1])
        axs[1].set_ylim([0, 500])

        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_fig(self, fig, path_to_save):
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        file_path = os.path.join(path_to_save, "histograms.png")
        fig.savefig(file_path)


if __name__ == '__main__':

    for i in range(2):
        for j in range(10):
            path = os.path.join('data_collection/town01_multimodality_t_intersection_simples', f'route_0{i}', f'ep_0{j}')
            ActionsHistogram().main(path)




    # path='data_collection/town01_fixed_route_without_trajectory/'
    # # os.makedirs(path, exist_ok=False)
    # for i in range(len(os.listdir(path))):
    #     if os.listdir(path)[i][-4:] != '.mp4':
    #         if i < 10:
    #             index_str = "0"+str(i)
    #         else:
    #             index_str = str(i)
    #     for j in range(len(os.listdir(path + f'route_{index_str}/'))):
    #         if os.listdir(path)[i][-4:] != '.mp4':
    #             route_path = path + f'route_{index_str}/ep_0{j}' 
    #             object = FrontCameraMovieMaker(path=route_path, name_index=str(i)+f'_ep_0{j}')
    #             object.save_record()

    # path='gail_experts_multi_sempahore/'
    # for i in range(len(os.listdir(path))):
    #     route_path = path + f'route_0{i}/ep_00/birdview_masks'
    #     object = BirdViewMovieMaker(path=route_path)
    #     object.save_record()