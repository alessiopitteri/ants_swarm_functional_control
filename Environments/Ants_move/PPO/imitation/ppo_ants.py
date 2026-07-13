import sys
import os
import scipy
import random
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
from Algorithms.ppo.PPO import *
import networkx as nx
import numpy as np

def dist_mod_8(a, b):
    return min((a - b) % 8, (b - a) % 8)


class PPO_ants(PPO):

    def __init__(self, env,
                 gamma, action_bound, actor_lr, critic_lr, batch_size, weight_decay, config_net_file,
                 clip_epsilon, entropy_coeff, policy_steps, learning_steps,
                 state_dim, action_dim):
        
        super().__init__(env,
                 gamma, action_bound, actor_lr, critic_lr, batch_size, weight_decay, config_net_file,
                 clip_epsilon, entropy_coeff, policy_steps, learning_steps,
                 state_dim, action_dim)
        
        self.k = 2

    def meshgrid(self, x, y):

        # Compute wrapped indices efficiently using NumPy
        x_offsets = (x + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Lx
        y_offsets = (y + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Ly

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    
    def get_state(self):
        
        states = []
        self.env.compute_density_spec()
        self.env.compute_density()

        for i in range(self.env.n_special):

            j = self.env.n+i

            xx, yy = self.meshgrid(self.env.positions[j, 0], self.env.positions[j, 1])
            state1 = self.env.pheromones[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            # state2 = self.env.rho[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            # state3 = self.env.rho_spec[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            state4 = np.zeros((1, self.env.vision, self.env.vision))
            h1, h2 = self.env.directions[self.env.head[j]]
            state4[0, 1+h1, 1+h2] = 1
            #s12 = np.concatenate((state1, state2), axis=0)
            #s123 = np.concatenate((s12, state3), axis=0)
            s = np.concatenate((state1, state4), axis=0)
            states.append(s)

        return np.array(states)

    
    def execute_action(self, actions):
        
        #old_head = self.env.head[self.env.n:].copy()
        self.env.head[self.env.n:] = actions
        
        # delta_head = [dist_mod_8(old_head[i], self.env.head[self.env.n:][i])/7 for i in range(self.env.n_special)]

        self.env.update()
        
        new_states = self.get_state().copy()  
        
        op = self.env.order_parameter_p()
        rewards = np.ones(self.env.n_special)*op

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done    
    