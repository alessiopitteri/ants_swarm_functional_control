import sys
import os
import scipy
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ppo.PPO import *
import networkx as nx
import numpy as np

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
        
        self.grid = gaussuian_filter(25, 0.9, 0)
        

    def meshgrid(self, x, y):

        # Compute wrapped indices efficiently using NumPy
        x_offsets = (x + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Lx
        y_offsets = (y + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Ly

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    
    def get_state(self):
        
        states = []

        for i in range(self.env.n_special):

            #self.env.compute_density_spec()
            xx, yy = self.meshgrid(self.env.positions[self.env.n+i, 0], self.env.positions[self.env.n+i, 1])
            state = self.grid[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            states.append(state)

        return np.array(states)


    def execute_action(self, actions):
        
        self.env.head[self.env.n:] = actions

        self.env.update()

        new_states = self.get_state().copy()       

        rewards = np.array([state[0,1,1] for state in new_states])

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done  
    