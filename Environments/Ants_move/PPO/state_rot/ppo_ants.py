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
    
    def rigid_rotate_matrix(self, matrices):
        N, C, _, _ = matrices.shape
        matrices_rotated = matrices.copy()
        indexes = self.env.head[self.env.n:]

        for i in range(N):
            for c in range(C):
                
                ind = indexes[i]
                if ind == 0:
                    continue  # No rotation needed

                # Get original matrix
                matrix = matrices[i, c]

                if ind == 1:
                    matrices_rotated[i, c] = np.array([
                        [matrix[1,0], matrix[0,0], matrix[0,1]],
                        [matrix[2,0], matrix[1,1], matrix[0,2]],
                        [matrix[2,1], matrix[2,2], matrix[1,2]]
                    ])

                elif ind == 2:
                    matrices_rotated[i, c] = np.array([
                        [matrix[2, 0], matrix[1, 0], matrix[0, 0]],
                        [matrix[2, 1], matrix[1, 1], matrix[0, 1]],
                        [matrix[2, 2], matrix[1, 2], matrix[0, 2]]
                    ])

                elif ind == 3:
                    matrices_rotated[i, c] = np.array([
                        [matrix[2, 1], matrix[2, 0], matrix[1, 0]],
                        [matrix[2, 2], matrix[1, 1], matrix[0, 0]],
                        [matrix[1, 2], matrix[0, 2], matrix[0, 1]]
                    ])
                elif ind == 4:
                    matrices_rotated[i, c] = np.array([
                        [matrix[2, 2], matrix[2, 1], matrix[2, 0]],
                        [matrix[1, 2], matrix[1, 1], matrix[1, 0]],
                        [matrix[0, 2], matrix[0, 1], matrix[0, 0]]
                    ])

                elif ind == 5:
                    matrices_rotated[i, c] = np.array([
                        [matrix[1, 2], matrix[2, 2], matrix[2, 1]],
                        [matrix[0, 2], matrix[1, 1], matrix[2, 0]],
                        [matrix[0, 1], matrix[0, 0], matrix[1, 0]]
                    ])
                elif ind == 6:
                    matrices_rotated[i, c] = np.array([
                        [matrix[0, 2], matrix[1, 2], matrix[2, 2]],
                        [matrix[0, 1], matrix[1, 1], matrix[2, 1]],
                        [matrix[0, 0], matrix[1, 0], matrix[2, 0]]
                    ])
                elif ind == 7:
                    matrices_rotated[i, c] = np.array([
                        [matrix[0, 1], matrix[0, 2], matrix[1, 2]],
                        [matrix[0, 0], matrix[1, 1], matrix[2, 2]],
                        [matrix[1, 0], matrix[2, 0], matrix[2, 1]]
                    ])
                    
        return matrices_rotated

    
    def get_state(self):
        
        states = []
        self.env.compute_density_spec()
        self.env.compute_density()

        for i in range(self.env.n_special):

            xx, yy = self.meshgrid(self.env.positions[self.env.n+i, 0], self.env.positions[self.env.n+i, 1])
            state1 = self.env.pheromones[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            state2 = self.env.rho[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            state3 = self.env.rho_spec[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            s12 = np.concatenate((state1, state2), axis=0)
            s = np.concatenate((s12, state3), axis=0)
            states.append(s)
        states = np.array(states)       

        return self.rigid_rotate_matrix(states)
    
    
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
        
        self.env.head[self.env.n:] = (self.env.head[self.env.n:] + actions) % 8

        self.env.update()
        
        new_states = self.get_state().copy()  

        s0 = self.env.avg_pher()

        mask = self.env.pheromones > (self.k*s0)
        
        G = create_graph_from_grid(mask)
        rews = [self.component_map_nodes(G).get((site[0], site[1])) for site in self.env.positions[self.env.n:]]
        rews = [item if item is not None else [0, 0] for item in rews]

        rewards = np.array([rew[0]*rew[1] for rew in rews])

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done    
    