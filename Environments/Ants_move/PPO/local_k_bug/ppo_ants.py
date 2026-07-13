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
                 state_dim, action_dim, clip_grad,
                 alpha):
        
        super().__init__(env,
                 gamma, action_bound, actor_lr, critic_lr, batch_size, weight_decay, config_net_file,
                 clip_epsilon, entropy_coeff, policy_steps, learning_steps,
                 state_dim, action_dim, clip_grad)
        
        self.k = 2
        self.alpha = alpha

    def create_graph_from_grid_local(self, grid):
        rows, cols = grid.shape
        G = nx.Graph()

        # Get all active nodes (grid == 1)
        nodes = np.argwhere(grid == 1)
        node_set = set(map(tuple, nodes))
        G.add_nodes_from(node_set)

        # Define the 8-connected neighborhood
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        for r, c in node_set:
            for dr, dc in directions:
                nr, nc = (r + dr), (c + dc)
                if nr < rows and nc < cols and nr > 0 and nc > 0:
                    if grid[nr, nc] == 1 :
                        neighbor = (nr, nc)
                        if neighbor in node_set:
                            G.add_edge((r, c), neighbor)

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
            h1 = self.env.directions[self.env.head[j]][0]
            h2 = self.env.directions[self.env.head[j]][1]
            state4[0, self.env.vision//2+h1, self.env.vision//2+h2] = 1
            s12 = np.concatenate((state1, state2), axis=0)
            s123 = np.concatenate((s12, state3), axis=0)
            s = np.concatenate((s123, state4), axis=0)
            states.append(s)

        return np.array(states)
    
    
    def component_map_nodes_degree(self, G):
        component_map = {}

        # initialize the component map with default values for the whole lattice of dimensions Lx and Ly
        component_map = {(x, y): [0, 0] for x in range(self.env.Lx) for y in range(self.env.Ly)}

        for component in nx.connected_components(G):
            component_nodes = len(component)
            size = component_nodes
            if component_nodes < 2:
                continue

            for node in component:
                degree = G.degree(node)
                component_map[node] = [size, degree]

        return component_map

    
    def reward(self, state, new_state, ind, c):
        s0 = state[ind, 0]
        s2 = state[ind, 2]
        ns0 = new_state[ind, 0]
        ns1 = new_state[ind, 1]
        ns2 = new_state[ind, 2]

        mean_s0 = np.mean(s0)
        grid = s0 > self.k * mean_s0
        G_t_size = np.count_nonzero(grid) + np.count_nonzero(s2)

        mean_ns0 = np.mean(ns0)
        grid_t1 = (ns0 > self.k * mean_ns0) | (ns2 > 0)

        G_t1 = self.create_graph_from_grid_local(grid_t1)

        g_t1 = 0
        central_node = (c, c)

        if central_node in G_t1:
            component = nx.node_connected_component(G_t1, central_node)
            g_t1 = len(component)
        else:
            g_t1 = 0


        degree = G_t1.degree(central_node) if central_node in G_t1 else 0
        inv_degree = 1 / degree if degree > 0 else 0

        return g_t1 / G_t_size + self.alpha*inv_degree 



    def execute_action(self, actions):
        
        states = self.get_state().copy()
        self.env.head[self.env.n:] = actions
        self.env.update()
        new_states = self.get_state().copy()

        rewards = [self.reward(states, new_states, j, self.env.vision//2) for j in range(self.env.n_special)]
        rewards = np.array(rewards)

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done    
    