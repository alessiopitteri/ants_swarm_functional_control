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
        
        self.k = 1
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

    
    def reward(self, new_state, ind):
        
        ns_pher = new_state[ind,0]
        ns0 = np.mean(new_state[ind,0])
        plus_id = ns_pher > ns0
        minus_id = ns_pher <= ns0
        n_plus = np.sum(plus_id)
        n_minus = np.sum(minus_id)

        ns_rho = new_state[ind,1]

        c = self.env.vision//2
        grid = plus_id.copy()
        grid[c, c] = 1
        central_node = (c, c)

        G_t1 = self.create_graph_from_grid_local(grid)
        component = nx.node_connected_component(G_t1, central_node)
        g_t1 = len(component)
        degree = G_t1.degree(central_node)

        component_id = np.zeros((self.env.vision, self.env.vision), dtype=bool)
        for node in component:
            r, c = node
            component_id[r, c] = True

        if degree <= 2 and degree > 0:
            
            condition1 = False
            condition2 = False
            
            if degree==2:
                neighbors = list(G_t1.neighbors(central_node))

                (r1, c1), (r2, c2) = neighbors

                condition1 = abs(r1 - r2) == 2
                condition2 = abs(c1 - c2) == 2
                
            aligned = condition1 or condition2 or degree==1

            if aligned:
        
                if n_plus == 0 or n_minus == 0:
                    return -1
                else:
                    mu_plus = np.sum(ns_rho[component_id])
                    mu_minus = np.sum(ns_rho) - mu_plus
                    local_rho = np.sum(ns_rho)

                    if local_rho == 0:
                        return -1
                    else:    
                        local_op = (mu_plus - mu_minus)/local_rho
                        reward = local_op * min(g_t1, self.env.vision)/self.env.vision
                        return reward
                    
            else:
                return -1
        else:
            return -1


    def execute_action(self, actions):
        
        self.env.head[self.env.n:] = actions
        self.env.update()
        new_states = self.get_state().copy()

        rewards = [self.reward(new_states, j) for j in range(self.env.n_special)]
        rewards = np.array(rewards)

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done    
    