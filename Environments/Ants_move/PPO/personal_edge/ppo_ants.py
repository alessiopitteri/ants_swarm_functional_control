import sys
import os
import scipy
import random
import networkx as nx
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
from Algorithms.ppo.PPO import *

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
        self.neighbor_map = self.precompute_neighbors(self.env.Lx, self.env.Ly)

    def precompute_neighbors(self, rows, cols):
        directions = np.array([[-1, -1], [-1, 0], [-1, 1],
                            [0, -1],           [0, 1],
                            [1, -1],  [1, 0],  [1, 1]])
        neighbor_map = {}
        for r in range(rows):
            for c in range(cols):
                neighbors = [((r + dr) % rows, (c + dc) % cols) for dr, dc in directions]
                neighbor_map[(r, c)] = neighbors
        return neighbor_map


    def create_graph_from_grid(self, grid):

        G = nx.Graph()
        nodes = np.argwhere(grid == 1)
        G.add_nodes_from(map(tuple, nodes))

        for r, c in nodes:
            for nr, nc in self.neighbor_map[(r, c)]:
                if grid[nr, nc] == 1:
                    G.add_edge((r, c), (nr, nc))

        return G
    

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
            state2 = self.env.rho[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            state3 = self.env.rho_spec[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
            state4 = np.zeros((1, self.env.vision, self.env.vision))
            h1, h2 = self.env.directions[self.env.head[j]]
            state4[0, 1+h1, 1+h2] = 1
            s12 = np.concatenate((state1, state2), axis=0)
            s123 = np.concatenate((s12, state3), axis=0)
            s = np.concatenate((s123, state4), axis=0)
            states.append(s)

        return np.array(states)
    
    
    def component_map_nodes_degree(self, G):
        component_map = {}
        # initialize the component map with default values for the whole lattice of dimensions Lx and Ly
        component_map = {(x, y): [0, 0, 0] for x in range(self.env.Lx) for y in range(self.env.Ly)}

        total_nodes = G.number_of_nodes()

        for component in nx.connected_components(G):
            component_nodes = len(component)
            size = component_nodes / total_nodes
            if component_nodes < 1:
                continue

            # compute the number of edges in the component
            # component_edges = sum(1 for u in component for v in G[u] if v in component) // 2
            degree_sum = sum(G.degree(node) for node in component)
    
            # Convert component set to array of coordinates
            coords = np.array(list(component))
            rows = coords[:, 0]
            cols = coords[:, 1]

            # Vectorized rho sum
            sum_rho = np.sum(self.env.rho[rows, cols])
            avg_rho = sum_rho / self.env.n

            for node in component:
                degree = G.degree(node)
                if degree == 0: 
                    component_map[node] = [size, avg_rho, 0]
                else:
                    component_map[node] = [size, avg_rho, 2*component_nodes/degree_sum]

        return component_map

        
    def reward(self, component_map, x, y):

        g = component_map.get((x, y))[0]
        p = component_map.get((x, y))[1]
        e = component_map.get((x, y))[2]

        r3 = g*e*p

        return r3


    def execute_action(self, actions):
        
        self.env.head[self.env.n:] = actions

        self.env.update()
        
        new_states = self.get_state().copy()  

        s0 = self.env.avg_pher()

        mask = self.env.pheromones > (self.k*s0)
        
        G = self.create_graph_from_grid(mask)
        component_map = self.component_map_nodes_degree(G)
        rewards = [self.reward(component_map, self.env.positions[self.env.n+i, 0],
                                              self.env.positions[self.env.n+i, 1]) for i in range(self.env.n_special)]
        rewards = np.array(rewards)

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done    
    