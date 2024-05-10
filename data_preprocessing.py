import os
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

    def __preprocess_front_images(self, images_array, eval):
        obs_front = images_array['central_rgb'] if eval else np.array([np.array(ele[0]['central_rgb']) for ele in images_array])
        obs_front = np.transpose(obs_front, (0, 2, 3, 1))
        obs_front = DataHandler().to_greyscale(obs_front)
        obs_front = DataHandler().normalizing(obs_front)
        obs_front = DataHandler().stack_with_previous(obs_front)

        obs_left = images_array['left_rgb'] if eval else np.array([np.array(ele[0]['left_rgb']) for ele in images_array])
        obs_left = np.transpose(obs_left, (0, 2, 3, 1))
        obs_left = DataHandler().to_greyscale(obs_left)
        obs_left = DataHandler().normalizing(obs_left)
        obs_left = DataHandler().stack_with_previous(obs_left)

        obs_right = images_array['right_rgb'] if eval else np.array([np.array(ele[0]['right_rgb']) for ele in images_array])
        obs_right = np.transpose(obs_right, (0, 2, 3, 1))
        obs_right = DataHandler().to_greyscale(obs_right)
        obs_right = DataHandler().normalizing(obs_right)
        obs_right = DataHandler().stack_with_previous(obs_right)
        
        obs_stack = np.concatenate((obs_left, obs_front, obs_right), axis=3)
        return obs_stack

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
    
    def preprocess_images(self, images_array, feature: str, eval=False):
        if feature == 'birdview':
            return self.__preprocess_birdview(images_array, eval)
        elif feature == 'human':
            return self.__preprocess_human_images(images_array)
        elif feature == 'front':
            return self.__preprocess_front_images(images_array, eval)
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


if __name__ == '__main__':

    path='gail_experts_multi_bruno_3_simples/'
    # os.makedirs(path, exist_ok=False)
    for i in range(len(os.listdir(path))):
        if os.listdir(path)[i][-4:] != '.mp4':
            if i < 10:
                index_str = "0"+str(i)
            else:
                index_str = str(i)
        for j in range(len(os.listdir(path + f'route_{index_str}/'))):
            if os.listdir(path)[i][-4:] != '.mp4':
                route_path = path + f'route_{index_str}/ep_0{j}' 
                object = FrontCameraMovieMaker(path=route_path, name_index=str(i)+f'_ep_0{j}')
                object.save_record()

    # path='gail_experts_multi_sempahore/'
    # for i in range(len(os.listdir(path))):
    #     route_path = path + f'route_0{i}/ep_00/birdview_masks'
    #     object = BirdViewMovieMaker(path=route_path)
    #     object.save_record()