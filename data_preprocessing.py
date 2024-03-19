import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

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

    def stack_with_previous(self, images_array):
        if len(images_array.shape) == 3:
            images_array = np.expand_dims(images_array, axis=-1)
        batch_size, height, width, channels = images_array.shape
        stacked_images = np.zeros((batch_size, height, width, channels * 4), dtype=images_array.dtype)

        for i in range(batch_size):
            if i < 3:
                    stacked_images[i, :, :, :] = np.concatenate([
                    images_array[i, :, :, :],
                    images_array[i, :, :, :],
                    images_array[i, :, :, :],
                    images_array[i, :, :, :]
                ], axis=-1)
            else:
                stacked_images[i, :, :, :] = np.concatenate([
                    images_array[i, :, :, :],
                    images_array[i - 1, :, :, :],
                    images_array[i - 2, :, :, :],
                    images_array[i - 3, :, :, :]
                ], axis=-1)

        return stacked_images
    
    def stack_with_previous_front(self, images_array, seq):
        pass


    def __preprocess_birdview(self, images_array, eval=False):
        obs = images_array['birdview'] if eval else np.array([np.array(ele[0]['birdview']) for ele in images_array])
        obs = np.transpose(obs, (0, 2, 3, 1))
        obs = DataHandler().to_greyscale(obs)
        obs = DataHandler().normalizing(obs)
        obs = DataHandler().stack_with_previous(obs)
        return obs
    
    def __preprocess_human_images(self, images_array):
        images = DataHandler().to_greyscale(images_array)
        images = DataHandler().normalizing(images)
        images = DataHandler().stack_with_previous(images)
        return images
    
    def __preprocess_front_images(self, images_array):
        images = DataHandler().to_greyscale(images_array)
        images = DataHandler().normalizing(images)
        images = DataHandler().stack_with_previous(images)
        return images
    
    def preprocess_images(self, images_array, feature: str, eval=False):
        if feature == 'birdview':
            return self.__preprocess_birdview(images_array, eval)
        elif feature == 'human':
            return self.__preprocess_human_images(images_array)
        elif feature == 'front':
            return self.__preprocess_front_images(images_array)
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