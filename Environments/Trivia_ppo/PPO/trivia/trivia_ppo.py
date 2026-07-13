import sys
import os
from Algorithms.ppo_GAE.PPO import *
import numpy as np

class PPO_trivia(PPO):

    def __init__(self, env,
                 gamma, action_bound, actor_lr, critic_lr, batch_size, weight_decay, config_net_file,
                 clip_epsilon, entropy_coeff, policy_steps, learning_steps,
                 state_dim, action_dim, clip_grad=False,
                 use_gae=False, lam=0.95):
        
        super().__init__(env,
                 gamma, action_bound, actor_lr, critic_lr, batch_size, weight_decay, config_net_file,
                 clip_epsilon, entropy_coeff, policy_steps, learning_steps,
                 state_dim, action_dim, clip_grad=clip_grad,
                 use_gae=use_gae, lam=lam)
        

    def get_state(self):

        return self.env.state.copy()

    def execute_action(self, action):
                
        self.env.update(action)

        reward = self.get_state()[:,0,1,1]

        return self.env.state, reward, self.env.done  

