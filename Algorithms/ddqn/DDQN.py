import sys
import os
from Algorithms.ddqn.networks import *
from RL_abstr import *
import yaml
import torch
import torch.nn.functional as F
import random
import numpy as np


class DDQN(RL):

    def __init__(self, env, replay_buffer,
                 gamma, lerning_rate, tau, batch_size, epsilon_decay, weight_decay, config_net_file,
                 target_upd_steps = 10, epsilon_min = 0.01, epsilon_start = 1):
        super().__init__(env)

        # buffer
        self.replay_buffer = replay_buffer

        # hyperparameters
        self.gamma = gamma
        self.critic_lr = lerning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.weight_decay = weight_decay

        # epsilon greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.target_upd_steps = target_upd_steps

        # config file for the networks
        self.config_net_file = config_net_file
        

    def load_config(self):
        with open(self.config_net_file, 'r') as f:
            return yaml.safe_load(f)

    def initialize_networks(self, config):

        critic_cfg = config["critic"]

        # Critic Network
        if critic_cfg["type"] == "mlp":
            critic = MLP_critic(config['critic'], self.replay_buffer.state_dim[0], self.replay_buffer.action_dim)

        elif critic_cfg["type"] == "cnn":
            critic = CNN_critic(config['critic'], self.replay_buffer.state_dim, self.replay_buffer.action_dim)
        
        else:
            raise ValueError("Critic network must be either MLP or CNN+MLP.")
        
        return critic
    
    def setup_networks(self):
        
        config = self.load_config()

        self.critic = self.initialize_networks(config)
        self.target_critic = self.initialize_networks(config)

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay)


    def policy(self, states, explore=True):
        
        self.critic.eval()

        if (random.random() < self.epsilon) and explore:
            return np.random.randint(0, self.replay_buffer.action_dim, size=(len(states)))
        else:
            state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            with torch.no_grad():
                return self.critic(state_tensor).argmax(axis=1).numpy()


    def training_step(self):
        
        self.critic.train()

        if self.replay_buffer.get_size() < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q values
        current_q_values = self.critic(states).gather(1, actions.long())

        # Double DQN: Get next action from policy_net, then evaluate in target_net
        next_actions = self.critic(new_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_critic(new_states).gather(1, next_actions.long()).detach()

        # Compute expected Q values
        expected_q_values = rewards + (self.gamma * next_q_values)# * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # Optimize the model
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        self.decay_epsilon()

        # Update target network periodically
        if self.env.step % self.target_upd_steps == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
    

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)


    def training_episode(self):

        self.env.reset_objects()
        self.env.transient()
        state = self.get_state().copy()

        reward_hist = []
        action_hist = []

        while not self.env.done:

            action = self.policy(np.array(state))
            new_state, reward, done = self.execute_action(action)

            for i in range(len(action)):
                self.replay_buffer.add_to_buffer(state[i], action[i], reward[i], new_state[i], done)
                        
            if self.replay_buffer.size >= self.batch_size:

                self.training_step()

                '''for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)'''

            reward_hist.append(reward)
            action_hist.append(action)

            state = new_state.copy()

        return reward_hist, action_hist
            

    def test_episode(self):

        self.env.reset_objects()
        self.env.transient()
        states = self.get_state().copy()

        reward_collection = []
        action_collection = []
        states_collection = [states.copy()]

        done = False

        while not done:
            
            actions = self.policy(states, explore=False)

            new_states, rewards, done = self.execute_action(actions)
            reward_collection.append(rewards)
            action_collection.append(actions.copy())
            states_collection.append(new_states.copy())
            states = new_states.copy()

        return reward_collection, action_collection, states_collection
    

    def save_weights(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.critic.state_dict(), os.path.join(folder_path, "critic.pth"))
        torch.save(self.target_critic.state_dict(), os.path.join(folder_path, "target_critic.pth"))
    
    def load_weights(self, folder_path):
        self.critic.load_state_dict(torch.load(os.path.join(folder_path, "critic.pth"), weights_only=True))
        self.target_critic.load_state_dict(torch.load(os.path.join(folder_path, "target_critic.pth"), weights_only=True))

    def compute_critic_weights(self):
        weights = [param.data.clone() for param in self.critic.parameters()]
        return weights