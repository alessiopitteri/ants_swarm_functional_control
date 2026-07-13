import sys
import os
from Algorithms.ddpg.networks import *
from RL_abstr import *
import yaml
import torch
import torch.nn.functional as F
import numpy as np


class DDPG(RL):

    def __init__(self, env, replay_buffer,
                 gamma, action_lower_bound, action_upper_bound, actor_lr, critic_lr, tau, batch_size, noise, weight_decay,
                   config_net_file):
        
        super().__init__(env)

        # replay buffer
        self.replay_buffer = replay_buffer

        # hyperparameters
        self.gamma = gamma
        self.action_lower_bound = action_lower_bound   
        self.action_upper_bound = action_upper_bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr 
        self.tau = tau
        self.noise = noise
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # config file for the networks
        self.config_net_file = config_net_file


    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


    def initialize_networks(self, config):
        actor_cfg = config["actor"]
        critic_cfg = config["critic"]

        # Actor Network
        if actor_cfg["type"] == "mlp":
            actor = MLP_actor(config['actor'], self.replay_buffer.state_dim[0], self.replay_buffer.action_dim)

        elif actor_cfg["type"] == "cnn":
            actor = CNN_actor(config['actor'], self.replay_buffer.state_dim, self.replay_buffer.action_dim)
            
        else:
            raise ValueError("Actor network must be either MLP or CNN+MLP.")

        # Critic Network
        if critic_cfg["type"] == "mlp":
            critic = MLP_critic(config['critic'], self.replay_buffer.state_dim[0], self.replay_buffer.action_dim)

        elif critic_cfg["type"] == "cnn":
            critic = CNN_critic(config['critic'], self.replay_buffer.state_dim, self.replay_buffer.action_dim)
        
        else:
            raise ValueError("Critic network must be either MLP or CNN+MLP.")
        
        return actor, critic
    

    def setup_networks(self):
            
        config = self.load_config(self.config_net_file)

        self.actor, self.critic = self.initialize_networks(config)
        self.target_actor, self.target_critic = self.initialize_networks(config)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay)


    def policy(self, states, test=False):
        
        #evaluation mode
        self.actor.eval()

        states = torch.FloatTensor(np.array(states))
        actions = self.actor(states).detach().numpy().squeeze(1)

        if not test:
            actions = actions + self.noise * np.random.randn(*actions.shape) 

        # if actions are in the range of [0, 1], scale them to the action space
        # actions = actions*(self.action_upper_bound - self.action_lower_bound) + self.action_lower_bound

        # if actions are in the range of [-1, 1], scale them to the action space
        actions = actions*(self.action_upper_bound - self.action_lower_bound)/2 + (self.action_upper_bound + self.action_lower_bound)/2
        
        return np.clip(actions, self.action_lower_bound, self.action_upper_bound)


    def training_step(self, freeze_critic=False):
        
        #training mode
        self.actor.train()
        
        if self.replay_buffer.get_size() < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.replay_buffer.sample(self.batch_size)

        if freeze_critic:
            pass
        
        else:
            # Compute target Q value
            next_actions = self.target_actor(new_states)
            target_Q = self.target_critic(new_states, next_actions)
            target_Q = rewards + self.gamma * target_Q.detach()

            # Update critic
            current_Q = self.critic(states, actions)

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def training_episode(self):

        self.env.reset_objects()
        self.env.transient()
        state = self.get_state().copy()

        reward_hist = []
        action_hist = []

        done = False

        while not done:

            action = self.policy(state)
            new_state, reward, done = self.execute_action(action)
            
            for i in range(len(action)):
                self.replay_buffer.add_to_buffer(state[i], action[i], reward[i], new_state[i], done)
                        
            if self.replay_buffer.size >= self.batch_size:

                self.training_step()

                # Update target networks
                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            
            reward_hist.append(reward)
            action_hist.append(action)

            state = new_state.copy()

        return reward_hist, action_hist
                 

    def test_episode(self):

        self.env.reset_objects()
        self.env.transient()
        state = self.get_state()

        reward_collection = []
        action_collection = []
        state_collection = [state.copy()]

        done = False

        while not done:

            action = self.policy(state, test=True)
            new_state, reward, done = self.execute_action(action)
            reward_collection.append(reward)
            action_collection.append(action.copy())
            state_collection.append(new_state.copy())
            state = new_state.copy()

        return reward_collection, action_collection, state_collection
    

    def simulation_episode(self, policy):
        
        self.env.reset_objects()
        self.env.transient()

        reward_collection = []
        done = False

        while not done:

            action = policy
            new_state, reward, done = self.execute_action(action)
            reward_collection.append(reward)

        return reward_collection


    def save_weights(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(folder_path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(folder_path, "critic.pth"))
        torch.save(self.target_actor.state_dict(), os.path.join(folder_path, "target_actor.pth"))
        torch.save(self.target_critic.state_dict(), os.path.join(folder_path, "target_critic.pth"))

    def load_weights(self, folder_path):
        self.actor.load_state_dict(torch.load(os.path.join(folder_path, "actor.pth"), weights_only=True))
        self.critic.load_state_dict(torch.load(os.path.join(folder_path, "critic.pth"), weights_only=True))
        self.target_actor.load_state_dict(torch.load(os.path.join(folder_path, "target_actor.pth"), weights_only=True))
        self.target_critic.load_state_dict(torch.load(os.path.join(folder_path, "target_critic.pth"), weights_only=True))

    def compute_actor_weights(self):
        weights = [param.data.clone() for param in self.actor.parameters()]
        return weights

    def compute_critic_weights(self):
        weights = [param.data.clone() for param in self.critic.parameters()]
        return weights
