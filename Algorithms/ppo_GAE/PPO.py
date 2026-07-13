import sys
import os
from Algorithms.ppo.networks import *
from RL_abstr import *
import yaml
import torch
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from torch.nn.utils import clip_grad_norm_

class PPO(RL):

    def __init__(self, env,
                 gamma, action_bound, actor_lr, critic_lr, batch_size, weight_decay, config_net_file,
                 clip_epsilon, entropy_coeff, policy_steps, learning_steps,
                 state_dim, action_dim, clip_grad=True,
                 use_gae=False, lam=0.95, norm_advantages=True):
        
        super().__init__(env)

        self.gamma = gamma
        self.action_bound = action_bound   
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size 
        self.weight_decay = weight_decay
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.policy_steps = policy_steps
        self.learning_steps = learning_steps
        self.env = env

        self.config_net_file = config_net_file

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_grad = clip_grad

        self.lam = lam
        self.use_gae = use_gae
        self.norm_advantages = norm_advantages
    
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    
    def initialize_networks(self, config):
        actor_cfg = config["actor"]
        critic_cfg = config["critic"]

        # Actor Network
        if actor_cfg["type"] == "mlp":
            actor = MLP_actor(config['actor'], self.state_dim[0], self.action_dim)

        elif actor_cfg["type"] == "cnn":
            actor = CNN_actor(config['actor'], self.state_dim, self.action_dim)
            
        else:
            raise ValueError("Actor network must be either MLP or CNN+MLP.")

        # Critic Network
        if critic_cfg["type"] == "mlp":
            critic = MLP_critic(config['critic'], self.state_dim[0], self.action_dim)

        elif critic_cfg["type"] == "cnn":
            critic = CNN_critic(config['critic'], self.state_dim, self.action_dim)
        
        else:
            raise ValueError("Critic network must be either MLP or CNN+MLP.")
        
        return actor, critic
    

    def setup_networks(self):
            
        config = self.load_config(self.config_net_file)

        self.actor, self.critic = self.initialize_networks(config)
        self.target_actor, self.target_critic = self.initialize_networks(config)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay)


    def policy(self, states):
        
        #evaluation mode
        self.actor.eval()

        # Convert state to tensor
        state_tensor = torch.FloatTensor(states)

        # Get action probabilities and value prediction from the agent.
        action_pred = self.actor(state_tensor)

        # Convert logits to probabilities
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        # action = np.argmax(action_prob.detach(), axis=1)
        
        # Convert action to numpy array
        action = action.detach().numpy()

        return action
    
    def calculate_advantages(self, returns, values):
        advantages = returns - values
        # Normalize the advantage
        advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages
    
    def calculate_gae(self, rewards, values, gamma, lam):
        
        # Ensure tensors
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)

        T, N = rewards.shape

        values = values.view(T + 1, N)  # Ensure values is [T+1, N]

        advantages = torch.zeros(T, N, dtype=torch.float32)
        gae = torch.zeros(N, dtype=torch.float32)

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        returns = advantages + values[:-1]  # G_t = A_t + V(s_t)

        return returns, advantages

    def calculate_surrogate_loss(
            self,
            actions_log_probability_old,
            actions_log_probability_new,
            advantages):
        advantages = advantages.detach()
        policy_ratio = (
                actions_log_probability_new - actions_log_probability_old
                ).exp()
        #Equation 3 comes into the play
        surrogate_loss_1 = policy_ratio * advantages
        surrogate_loss_2 = torch.clamp(
                policy_ratio, min=1.0-self.clip_epsilon, max=1.0+self.clip_epsilon
                ) * advantages
        surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        return surrogate_loss

    def calculate_losses(
            self,
            surrogate_loss, entropy, returns, value_pred):
        # We calulate entropy and total policy by equation 2 and 4
        entropy_bonus = self.entropy_coeff * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).sum()
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
        return policy_loss, value_loss

    '''def calculate_returns(rewards, gamma):
        # Returns are just discounted rewards as was visible in equation 1
        returns = torch.zeros_like(torch.FloatTensor(np.array(rewards)))
        returns[-1] = rewards[-1]
        for i in reversed(range(len(rewards) - 1)):
            returns[i] = rewards[i] + gamma * returns[i + 1]
        return returns'''
    
    def calculate_returns(self, rewards, gamma):
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)  # Convert to tensor
        T, N = rewards.shape  # Get dimensions

        # Create discount coefficients with the correct shape [T, 1] to match broadcasting
        discounts = torch.full((T, 1), gamma, dtype=torch.float32).cumprod(dim=0)

        # Compute discounted sum using cumulative sum trick
        returns = torch.flip(torch.cumsum(torch.flip(rewards * discounts, [0]), dim=0), [0]) / discounts

        return returns


        # Update policy and value networks using PPO.
    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        
        # Set the agent to training mode.
        self.actor.train()
        self.critic.train()

        # Create DataLoader for mini-batches
        dataset = DataLoader(
            TensorDataset(states, actions, old_log_probs.detach(), advantages, returns),
            self.batch_size, shuffle=True
        )

        for _ in range(self.learning_steps):
            for batch in dataset:
                batch_states, batch_actions, old_probs, adv, ret = batch
                action_pred = self.actor(batch_states)
                value_pred = self.critic(batch_states)
                value_pred = value_pred.squeeze(-1)

                # Calculate new action probabilities and entropy.
                action_prob = F.softmax(action_pred, dim=-1)
                dist = distributions.Categorical(action_prob)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # Calculate policy loss (surrogate loss) and value loss.
                surrogate_loss = self.calculate_surrogate_loss(old_probs, new_log_probs, adv)
                policy_loss, value_loss = self.calculate_losses(surrogate_loss, entropy, ret, value_pred)

                # Backpropagate and update weights.
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                if self.clip_grad:
                    clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                if self.clip_grad:
                    clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

    def forward_pass(self, steps):

        states, actions, log_probs, values, rewards, done, episode_reward = [], [], [], [], [], False, 0.0
        time_steps = 0
        state = self.get_state().copy()

        while time_steps in range(steps) and not done:
            state_tensor = torch.FloatTensor(state)#.unsqueeze(0)
            states.append(state_tensor)

            # Get action probabilities and value prediction from the agent.
            action_pred = self.actor(state_tensor)
            value_pred = self.critic(state_tensor)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()

            # Store log probability of the selected action.
            log_probs.append(dist.log_prob(action))
            values.append(value_pred)
            new_state, reward, done = self.execute_action(action.detach().numpy())
            
            rewards.append(reward)
            episode_reward += np.mean(reward)
            actions.append(action)

            state = new_state.copy()
            time_steps += 1
        
        values.append(self.critic(torch.FloatTensor(state)))  # Get value for the last state

        if self.use_gae:
            # Convert to tensors
            states = torch.cat(states)
            log_probs, values = torch.cat(log_probs), torch.cat(values).squeeze(-1)
            actions = torch.cat(actions)

            # GAE estimation
            returns, advantages = self.calculate_gae(rewards, values, self.gamma, self.lam)
            # Normalize advantages
            returns = returns.view(-1).detach()
            values = values.view(-1).detach()
            advantages = advantages.view(-1).detach()
            if self.norm_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        else:
            # Convert to tensors
            states = torch.cat(states)
            log_probs, values = torch.cat(log_probs), torch.cat(values).squeeze(-1)
            actions = torch.cat(actions)

            # Standard Monte Carlo estimation
            returns = self.calculate_returns(rewards, self.gamma)
            returns = returns.view(-1)
            
            advantages = returns - values[:-1]

        return episode_reward, states, actions, log_probs, advantages, returns, rewards

    def training_episode(self):

        self.env.reset_objects()
        self.env.transient()
        self.actor.train()
        self.critic.train()

        accumulated_reward = []

        while not self.env.done:
            #print('forward pass')
            episode_reward, states, actions, log_probs, advantages, returns, rewards = self.forward_pass(self.policy_steps)
            #print('update policy')
            self.update_policy(states, actions, log_probs, advantages, returns)
            accumulated_reward.append(rewards)
            #print('end')
        rewards = np.array(accumulated_reward).reshape(-1, len(accumulated_reward[0]))
        rewards = np.mean(rewards, axis=1)

        return rewards, actions.detach().numpy()
    

    def test_forward_pass(self, max_prob=False, save_pos = False, reset=True):

        states, actions, rewards, done, episode_reward = [], [], [], False, 0.0
        if reset:
            self.env.reset_objects()
        self.env.transient()
        self.actor.eval()
        self.critic.eval()
        
        state = self.get_state().copy()
        state_tensor = torch.FloatTensor(state)

        if save_pos:
            pos = []
            pher = []

        while not done:
            # Get action probabilities and value prediction from the agent.
            action_pred = self.actor(state_tensor)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            
            if max_prob:
                action = np.argmax(action_prob.detach(), axis=1)

            # Store log probability of the selected action.
            new_state, reward, done = self.execute_action(action.detach().numpy())
            
            rewards.append(reward)
            episode_reward += np.mean(reward)
            actions.append(action)
            if save_pos:
                pos.append(self.env.positions.copy())
                pher.append(self.env.pheromones.copy())

            state = new_state.copy() 
            state_tensor = torch.FloatTensor(state)
            states.append(state_tensor)


        # Convert to tensors and calculate advantages (returns - values).
        states = torch.cat(states)
        actions = torch.cat(actions)

        if save_pos: 
            return episode_reward, states, actions, rewards, pos, pher
        
        return episode_reward, states, actions, rewards

    def test_episode(self, max_prob=False, save_pos = False, reset=True):

        if save_pos:
            episode_reward, states, actions, rewards, pos, pher = self.test_forward_pass(max_prob, save_pos=save_pos, reset=reset)
            return rewards, actions.detach().numpy(), states, pos, pher

        else:
            episode_reward, states, actions, rewards = self.test_forward_pass(max_prob, save_pos=save_pos, reset=reset)
            return rewards, actions.detach().numpy()


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

    def load_weights(self, folder_path):
        self.actor.load_state_dict(torch.load(os.path.join(folder_path, "actor.pth"), weights_only=True))
        self.critic.load_state_dict(torch.load(os.path.join(folder_path, "critic.pth"), weights_only=True))

    def compute_actor_weights(self):
        weights = [param.data.clone() for param in self.actor.parameters()]
        return weights

    def compute_critic_weights(self):
        weights = [param.data.clone() for param in self.critic.parameters()]
        return weights