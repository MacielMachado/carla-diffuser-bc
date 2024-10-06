from models_bc import Model_cnn_BC, Model_cnn_BC_resnet, Model_cnn_GKC
from data_preprocessing import DataHandler, CarlaCustomDataset, CarlaCustomDatasetSpeed
from expert_dataset import ExpertDataset
from torchvision import transforms
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import torch
import wandb
import gym
import git
import os


class TrainerSemaphores():
    def __init__(self, n_epoch, lrate, device, n_hidden, batch_size, n_T,
                 net_type, drop_prob, extra_diffusion_steps, embed_dim,
                 guide_w, betas, dataset_path, run_wandb, record_run,
                 expert_dataset, name='', param_search=False,
                 embedding="Model_cnn_mlp", observation_type="birdview", use_velocity=False):

        self.n_epoch = n_epoch
        self.lrate = lrate
        self.device = device
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.n_T = n_T
        self.net_type = net_type
        self.drop_prob = drop_prob
        self.extra_diffusion_steps = extra_diffusion_steps
        self.embed_dim = embed_dim
        self.guide_w = guide_w
        self.betas = betas
        self.dataset_path = dataset_path
        self.name = name
        self.param_search = param_search
        self.run_wandb = run_wandb
        self.record_rum = record_run
        self.embedding = embedding
        self.best_reward = float('-inf')
        self.patience = 20
        self.early_stopping_counter = 0
        self.expert_dataset = expert_dataset
        self.observation_type = observation_type
        self.use_velocity = use_velocity

    def main(self):
        if self.run_wandb:
            self.config_wandb(project_name="Carla-Diffuser-Multi-Front-Speed",
                              name=self.name)
        dataload_train = self.prepare_dataset(self.expert_dataset)
        x_dim, y_dim = self.get_x_and_y_dim(dataload_train)
        model = self.create_conv_model(x_dim, y_dim)
        optim = self.create_optimizer(model)
        model = self.train(model, dataload_train, optim)

    def config_wandb(self, project_name, name):
        wandb.login(key='9bcc371f01af2fc8ddab2c3ad226caad57dc4ac5')
        config={
                "n_epoch": self.n_epoch,
                "lrate": self.lrate,
                "device": self.device,
                "n_hidden": self.n_hidden,
                "batch_size": self.batch_size,
                "n_T": self.n_T,
                "net_type": self.net_type,
                "drop_prob": self.drop_prob,
                "extra_diffusion_steps": self.extra_diffusion_steps,
                "embed_dim": self.embed_dim,
                "guide_w": self.guide_w,
                "dataset": self.dataset_path,
                "model": self.embedding,
                "commit_hash": self.get_git_commit_hash()
            }
        if name != '':
            return wandb.init(project=project_name, name=name, config=config)
        return wandb.init(project=project_name, config=config)

    def get_git_commit_hash(self):
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha

    def prepare_dataset(self, dataset):
        # Actions
        actions = np.array([np.array(ele[0]['actions']) for index, ele in enumerate(dataset)])
        # Previous Actions
        previous_actions = np.empty_like(actions)
        previous_actions[:-1], previous_actions[-1] = actions[1:], actions[-1] 
        # Observation
        state = np.array([np.array(ele[0]['state']) for index, ele in enumerate(dataset)])
        # Speed
        # speed = state[:5,-2:]
        speed = state[:,[4,5]]
        obs = DataHandler().preprocess_images(
            dataset,
            observation_type=self.observation_type,
            stack_with_previous=not self.use_velocity,
            use_velocity=self.use_velocity,
            use_greyscale=False)
        
        if self.use_velocity:
            dataset = CarlaCustomDatasetSpeed(obs, actions, speed, previous_actions)
        else:
            dataset = CarlaCustomDataset(obs, actions)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        '''
        The datasets have keys with the following information: birdview,
        central_rgb, left_rgb, right_rgb, item_idx, done, action, state
        '''
        return dataloader
    
        # obs_path = self.expert_dataset+'all_observations.pth'
        # obs = np.array(torch.load(obs_path))
        # obs = np.transpose(obs, (0,1,3,4,2))
        # obs = DataHandler().preprocess_images(obs, feature='front')
        # # obs = cv2.resize(obs[0], dsize=(96, 96), interpolation=cv2.INTER_CUBIC)[:,:,0], cmap=plt.get_cmap("gray")
        # state = np.array([np.array(ele[0]['state']) for ele in dataset])
        # actions = np.array([np.array(ele[0]['actions']) for ele in dataset])
        # dataset = CarlaCustomDataset(obs, actions)
        # dataloader = data.DataLoader(dataset,
        #                              batch_size=self.batch_size,
        #                              shuffle=True)
        # '''
        # The datasets have keys with the following information: birdview,
        # central_rgb, left_rgb, right_rgb, item_idx, done, action, state
        # '''
        # return dataloader
    
    def get_x_and_y_dim(self, dataset):
        '''
        '''
        y_dim = tuple(next(iter(dataset))[1].shape)[-1]
        x_dim = tuple(next(iter(dataset))[0].shape)[1:]
        return x_dim, y_dim
    
    def create_conv_model(self, x_dim, y_dim):
        cnn_out_dim = 4608
        if self.use_velocity:
            cnn_out_dim = 512 + 2

        if self.embedding == "Model_cnn_bc":
            return Model_cnn_bc(self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type).to(self.device)
        elif self.embedding == "Model_cnn_mlp":
            return Model_cnn_mlp(x_dim, self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type,
                                cnn_out_dim=cnn_out_dim).to(self.device)
        elif self.embedding == "Model_cnn_mlp_speed":
            return Model_cnn_mlp_speed(x_dim, self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type,
                                cnn_out_dim=cnn_out_dim,
                                use_velocity=self.use_velocity).to(self.device)
        elif self.embedding == "Model_cnn_mlp_GKC":
            return Model_cnn_GKC(x_dim, y_dim, self.embed_dim,
                                 self.net_type,
                                 embed_n_hidden=self.n_hidden).to(self.device)
        else:
            raise NotImplementedError
    
    def create_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lrate)
    
    def create_agent_model(self, conv_model, x_dim, y_dim):
        return Model_Cond_Diffusion(
            conv_model,
            betas=self.betas,
            n_T = self.n_T,
            device=self.device,
            x_dim=x_dim,
            y_dim=y_dim,
            drop_prob=self.drop_prob,
            guide_w=self.guide_w
        ).to(self.device)

    def decay_lr(self, epoch, lrate):
        return lrate * ((np.cos((epoch / self.n_epoch) * np.pi) + 1) / 2)
    
    def train(self, model, dataload_train, optim):
        for ep in tqdm(range(self.n_epoch), desc="Epoch"):
            results_ep = [ep]
            model.train()

            lr_decay = self.decay_lr(ep, self.lrate)
            # train loop
            pbar = tqdm(dataload_train)
            loss_ep, n_batch = 0, 0
            for batch in pbar:
                x_batch = batch[0].type(torch.FloatTensor).to(self.device)
                y_batch = batch[1].type(torch.FloatTensor).to(self.device)
                speed = batch[2].type(torch.FloatTensor).to(self.device) if self.use_velocity else None
                previous_actions = batch[3].type(torch.FloatTensor).to(self.device) if self.use_velocity else None
                y_hat = model(x_batch, speed, previous_actions)
                loss = self.loss_func(y_hat, y_batch)
                optim.zero_grad()
                loss.backward()
                loss_ep += loss.detach().item()
                n_batch += 1
                pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
                optim.step()

                with torch.no_grad():
                    y_hat_batch = model(x_batch, speed, previous_actions)
                    action_MSE = extract_action_mse(y_batch, y_hat_batch)

                if self.run_wandb:
                    # log metrics to wandb
                    wandb.log({"loss": loss_ep/n_batch,
                                "lr": lr_decay,
                                "steering_MSE": action_MSE[0],
                                "acceleration_MSE": action_MSE[1]})
                        
                    results_ep.append(loss_ep / n_batch)

            if ep in [1, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 350, 400, 500, 600, 750, 850, 950, 1000]:
                name=f'model_novo_ep_{ep}'
                self.save_model(model, ep)

        if self.run_wandb:
            wandb.finish()
        
        return model

    def loss_func(self, y, y_hat):
        return torch.nn.MSELoss()(y, y_hat)

    def save_model(self, model, ep=''):
        os.makedirs(os.getcwd()+'/model_pytorch/multi/BC/BC_Multi_GKC/'+self.name, exist_ok=True)
        torch.save(model.state_dict(), os.getcwd()+'/model_pytorch/multi/BC/BC_Multi_GKC/'+self.name+'_'+self.get_git_commit_hash()[0:4]+'_ep_'+f'{ep}'+'.pkl')


def extract_action_mse(y, y_hat):
    assert len(y) == len(y_hat)
    y_diff_pow_2 = torch.pow(y - y_hat, 2)
    y_diff_sum = torch.sum(y_diff_pow_2, dim=0)/len(y)
    mse = torch.pow(y_diff_sum, 0.5)
    return mse


if __name__ == '__main__':
    resume_last_train = False
    observation_space = {}
    observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)  # Define o tipo de dado que tará uma dimensão
    observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)  # Define o tipo de dimensão que terá o estado
    observation_space = gym.spaces.Dict(**observation_space)  # Cria um espaço de observação
    action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)  # Define o espaço de ação
    device = 'cuda'
    batch_size = 24

    '''
    The datasets have keys with the following information: birdview,
    central_rgb, left_rgb, right_rgb, item_idx, done, action, state
    '''
    stop  = 1

    # Dataset
    dataset_path = "/home/casa/projects/bruno/carla-diffuser-bc/bet_data_release/carla/"
    # obs = torch.load("/home/casa/projects/bruno/carla-diffuser-bc/bet_data_release/carla/all_observations.pth")
    # actions = torch.load("/home/casa/projects/bruno/carla-diffuser-bc/bet_data_release/carla/all_actions_pm1.pth")
    # seq = torch.load("/home/casa/projects/bruno/carla-diffuser-bc/bet_data_release/carla/seq_lengths.pth")

    # zero_index = np.where(np.array(seq) == 0)
    # obs = [np.array(ele[0:max_index, :, :, :]) for ele, max_index in zip(obs, seq)]
    # actions = [np.array(ele[0:max_index, :, :, :]) for ele, max_index in zip(actions, seq)]

    TrainerSemaphores(
        n_epoch=750,
        lrate=0.0001,
        device='cpu', 
        n_hidden=128,
        batch_size=32,
        n_T=20,
        net_type='transformer',
        drop_prob=0.0,
        extra_diffusion_steps=16,
        embed_dim=128,
        guide_w=0.0,
        betas=(1e-4, 0.02),
        dataset_path='data_collection/town01_multimodality_t_intersection_simples',
        run_wandb=True,
        record_run=True,
        expert_dataset=ExpertDataset('data_collection/town01_multimodality_t_intersection_simples', n_routes=2, n_eps=10, semaphore=False),
        name='town01_BC_multi_without_trajectory_birdview_GKC_speed',
        param_search=False,
        embedding="Model_cnn_mlp_GKC",
        observation_type='birdview',
        use_velocity=True).main()






