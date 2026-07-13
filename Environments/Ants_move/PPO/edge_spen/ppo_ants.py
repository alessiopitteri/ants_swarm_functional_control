import sys
import os
import scipy
import random
import networkx as nx
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
from Algorithms.ppo_GAE.PPO import *

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
                 state_dim, action_dim, clip_grad=False,
                 use_gae=False, lam=0.95, norm_advantages=True):
        
        super().__init__(env,
                 gamma, action_bound, actor_lr, critic_lr, batch_size, weight_decay, config_net_file,
                 clip_epsilon, entropy_coeff, policy_steps, learning_steps,
                 state_dim, action_dim, clip_grad,
                 use_gae=use_gae, lam=lam, norm_advantages=norm_advantages)
        
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

        c = self.env.vision//2
        grid = plus_id.copy()
        grid[c, c] = 1
        central_node = (c, c)

        G_t1 = self.create_graph_from_grid_local(grid)
        component = nx.node_connected_component(G_t1, central_node)
        n_nodes = len(component)
        # compute the number of edges in the CC
        n_edges = G_t1.subgraph(component).number_of_edges()

        if n_nodes == 1:
            edge_pen = 0
        else:
            edge_pen = -n_edges/(n_nodes-1)

        #mean_degree = np.mean([d for n, d in G_t1.subgraph(component).degree()])

        ns_rho = new_state[ind,1]

        pop = 0
        pod = 0

        component_id = np.zeros((self.env.vision, self.env.vision), dtype=bool)
        for node in component:
            r, c = node
            component_id[r, c] = True

        if n_plus == 0 or n_minus == 0:
            pod = -1
            pop = 0
        else:
            sigma_plus = np.mean(ns_pher[component_id])
            sigma_minus = np.mean(ns_pher[~component_id])
            sigma_tot = sigma_plus + sigma_minus

            mu_plus = np.sum(ns_rho[component_id])
            mu_minus = np.sum(ns_rho) - mu_plus
            local_rho = np.sum(ns_rho)

            if local_rho == 0:
                pod = -1
            else:    
                pod = (mu_plus - mu_minus)/local_rho

            if sigma_tot == 0:
                pop = 0
            else:    
                pop = (sigma_plus - sigma_minus)/sigma_tot

        return edge_pen + (pop*min(n_nodes,self.env.vision)/self.env.vision+pod)/2


    def execute_action(self, actions):
        
        self.env.head[self.env.n:] = actions
        self.env.update()
        new_states = self.get_state().copy()

        rewards = [self.reward(new_states, j) for j in range(self.env.n_special)]
        rewards = np.array(rewards)

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done    
    

    def global_meshgrid(self, x, y):

        # Compute wrapped indices efficiently using NumPy
        x_offsets = (x + np.arange(-(self.env.LL//2), (self.env.LL//2)+1)) % self.env.LL
        y_offsets = (y + np.arange(-(self.env.LL//2), (self.env.LL//2)+1)) % self.env.LL

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    

    def get_global_state(self):
        
        states = []
        self.env.compute_density_spec()
        self.env.compute_density()

        for i in range(self.env.n_special):

            j = self.env.n+i

            xx, yy = self.global_meshgrid(self.env.positions[j, 0], self.env.positions[j, 1])
            state1 = self.env.pheromones[xx, yy].reshape(1,self.env.LL,self.env.LL).copy()
            state2 = self.env.rho[xx, yy].reshape(1,self.env.LL,self.env.LL).copy()
            state3 = self.env.rho_spec[xx, yy].reshape(1,self.env.LL,self.env.LL).copy()
            state4 = np.zeros((1, self.env.LL, self.env.LL))
            h1 = self.env.directions[self.env.head[j]][0]
            h2 = self.env.directions[self.env.head[j]][1]
            state4[0, self.env.LL//2+h1, self.env.LL//2+h2] = 1
            s12 = np.concatenate((state1, state2), axis=0)
            s123 = np.concatenate((s12, state3), axis=0)
            s = np.concatenate((s123, state4), axis=0)
            states.append(s)

        return np.array(states)
    
    
    def global_reward(self):
        
        ns_pher = self.env.pheromones.copy()
        ns0 = np.mean(ns_pher)
        plus_id = ns_pher > ns0
        grid = plus_id.copy()

        G_t1 = self.create_graph_from_grid_local(grid)
        components = list(nx.connected_components(G_t1))

        metrics_for_components = []        

        for component in components:
            n_nodes = len(component)
            if n_nodes < 2:
                continue
            n_edges = G_t1.subgraph(component).number_of_edges()
            edge_pen = -n_edges/(n_nodes-1)

            pop = self.env.order_parameter_s()
            pod = self.env.order_parameter_p()

            len_rew = min(n_nodes,self.env.Lx)/self.env.Lx
            rew = edge_pen + (pop*len_rew+pod)/2
        
            metrics_for_components.append([n_edges, n_nodes, rew])

        return np.array(metrics_for_components)