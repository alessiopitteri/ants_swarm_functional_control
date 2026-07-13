import sys
import os
import scipy
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ppo.PPO import *
import networkx as nx
import numpy as np
from utils import angle_btw

def rotate_array(arr, indices):
    n = arr.shape[0]
    return np.array([np.roll(arr, -i % n) for i in indices])

def create_graph_from_grid(grid):
        rows, cols = grid.shape
        G = nx.Graph()

        # Get all active nodes (where grid == 1)
        nodes = np.argwhere(grid == 1)
        G.add_nodes_from(map(tuple, nodes))  # Convert to tuple for hashability

        # Define the 8 possible moves
        directions = np.array([[-1, -1], [-1, 0], [-1, 1],
                               [0, -1],            [0, 1],
                               [1, -1],  [1, 0],  [1, 1]])

        # Compute valid neighbor coordinates with periodic boundaries
        for r, c in nodes:
            for dr, dc in directions:
                nr = (r + dr) % rows  # Wrap around row boundaries
                nc = (c + dc) % cols  # Wrap around column boundaries
                if grid[nr, nc] == 1:  # Only add edges if the neighbor is also 1
                    G.add_edge((r, c), (nr, nc))

        return G 

def get_largest_connected_component(G):
    try:
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()
    except ValueError:  
        return nx.Graph() 
    
def gaussuian_filter(kernel_size, sigma=1, muu=0):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-2, 2, kernel_size),
                       np.linspace(-2, 2, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1.0/(2* np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal

    return gauss

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
        self.grid = gaussuian_filter(25, 0.9, 0)


    def get_state(self):
        
        self.env.compute_density_spec()
        self.env.compute_density()

        # Vectorized indices calculation for new positions
        indices = (self.env.positions[self.env.n:, None, :] + self.env.directions) % np.array([self.env.Lx, self.env.Ly])
        
        # Extract pheromone values at new positions
        pheromones = self.env.pheromones[indices[:, :, 0], indices[:, :, 1]]

        state = np.concatenate((pheromones, self.env.head[self.env.n:].reshape(-1,1)), axis=1)

        return state
    
    
    def component_map_nodes(self, G):
        component_map = {}
        total_nodes = G.number_of_nodes()

        for component in nx.connected_components(G):
            size = len(component) / total_nodes
            if size == 0:
                continue

            # Convert component set to array of coordinates
            coords = np.array(list(component))
            rows = coords[:, 0]
            cols = coords[:, 1]

            # Vectorized rho sum
            sum_rho = np.sum(self.env.rho[rows, cols])
            avg_rho = sum_rho / self.env.n

            for node in component:
                component_map[node] = [size, avg_rho]

        return component_map


    def execute_action(self, actions):

        # get head indexes 
        self.env.head[self.env.n:] = (self.env.head[self.env.n:] + actions) % 8

        self.env.update()

        new_states = self.get_state().copy()  

        s0 = self.env.avg_pher()

        mask = self.env.pheromones > (self.k*s0)
        
        '''G = create_graph_from_grid(mask)
        rews = [self.component_map_nodes(G).get((site[0], site[1])) for site in self.env.positions[self.env.n:]]
        rews = [item if item is not None else [0, 0] for item in rews]

        rewards = np.array([rew[0]*rew[1] for rew in rews])'''

        rewards = np.array([self.grid[self.env.positions[self.env.n+i,0],
                                      self.env.positions[self.env.n+i,1]] for i in range(self.env.n_special)])

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done
    