from models import Model_cnn_bc, Model_cnn_mlp, Model_Cond_Diffusion
from data_preprocessing import DataHandler, CarlaCustomDataset
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


class Trainer():
    def __init__(self, n_epoch, lrate, device, n_hidden, batch_size, n_T,
                 net_type, drop_prob, extra_diffusion_steps, embed_dim,
                 guide_w, betas, dataset_path, run_wandb, record_run,
                 expert_dataset, name='', param_search=False,
                 embedding="Model_cnn_mlp",):

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

    def main(self):
        if self.run_wandb:
            self.config_wandb(project_name="Carla-Diffuser", name=self.name)
        dataload_train = self.prepare_dataset(self.expert_dataset)
        x_dim, y_dim = self.get_x_and_y_dim(dataload_train)
        conv_model = self.create_conv_model(x_dim, y_dim)
        model = self.create_agent_model(conv_model, x_dim, y_dim)
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

    def prepare_dataset(self, dataset):
        obs = DataHandler().preprocess_images(dataset, feature='birdview')
        # obs = cv2.resize(obs[0], dsize=(96, 96), interpolation=cv2.INTER_CUBIC)[:,:,0], cmap=plt.get_cmap("gray")
        state = np.array([np.array(ele[0]['state']) for ele in dataset])
        actions = np.array([np.array(ele[0]['actions']) for ele in dataset])
        dataset = CarlaCustomDataset(obs, actions)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        '''
        The datasets have keys with the following information: birdview,
        central_rgb, left_rgb, right_rgb, item_idx, done, action, state
        '''
        return dataloader
    
    def get_x_and_y_dim(self, dataset):
        '''
        '''
        y_dim = tuple(next(iter(dataset))[1].shape)[-1]
        x_dim = tuple(next(iter(dataset))[0].shape)[1:]
        return x_dim, y_dim
    
    def create_conv_model(self, x_dim, y_dim):
        cnn_out_dim = 4608
        if self.embedding == "Model_cnn_bc":
            return Model_cnn_bc(self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type).to(self.device)
        elif self.embedding == "Model_cnn_mlp":
            return Model_cnn_mlp(x_dim, self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type,
                                cnn_out_dim=cnn_out_dim).to(self.device)
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
            for x_batch, y_batch in pbar:
                x_batch = x_batch.type(torch.FloatTensor).to(self.device)
                y_batch = y_batch.type(torch.FloatTensor).to(self.device)
                loss = model.loss_on_batch(x_batch, y_batch)
                optim.zero_grad()
                loss.backward()
                loss_ep += loss.detach().item()
                n_batch += 1
                pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
                optim.step()

                with torch.no_grad():
                    y_hat_batch = model.sample(x_batch)
                    action_MSE = extract_action_mse(y_batch, y_hat_batch)

                if self.run_wandb:
                    # log metrics to wandb
                    wandb.log({"loss": loss_ep/n_batch,
                                "lr": lr_decay,
                                "steering_MSE": action_MSE[0],
                                "acceleration_MSE": action_MSE[1]})
                        
                    results_ep.append(loss_ep / n_batch)

            if ep in [1, 20, 40, 80, 150, 250, 500, 600, 749]:
                name=f'model_novo_ep_{ep}'
                self.save_model(model, ep)

        if self.run_wandb:
            wandb.finish()
        
        return model


    def save_model(self, model, ep=''):
        os.makedirs(os.getcwd()+'/model_pytorch/'+self.name, exist_ok=True)
        torch.save(model.state_dict(), os.getcwd()+'/model_pytorch/'+self.name+'_'+self.get_git_commit_hash()[0:4]+'_ep_'+f'{ep}'+'.pkl')

env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}


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
    device = 'cpu'
    batch_size = 24

    gail_train_loader = torch.utils.data.DataLoader(
        ExpertDataset('gail_experts', n_routes=1, n_eps=1),
        batch_size=batch_size,
        shuffle=True,
    )

    '''
    The datasets have keys with the following information: birdview,
    central_rgb, left_rgb, right_rgb, item_idx, done, action, state
    '''
    stop  = 1

    Trainer(n_epoch=750,
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
            dataset_path='gail_experts',
            run_wandb=True,
            record_run=True,
            expert_dataset=ExpertDataset('gail_experts', n_routes=1, n_eps=1),
            name='gail_experts_nroutes1_neps1',
            param_search=False,
            embedding="Model_cnn_mlp",).main()





