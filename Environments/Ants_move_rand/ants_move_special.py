import numpy as np
from collections import deque
from utils import *
import random
import pygame
import networkx as nx

# weight function
def phr_weight(sigma, delta, beta):
    return (1 + sigma/(1+delta*sigma))**beta

def compute_probability(args):
    i, pheromones, delta, beta, w = args
    p_i = phr_weight(pheromones[i], delta[i], beta[i]) * w[i]
    p_i /= p_i.sum()  # Normalize probabilities
    return p_i

class Ants_move_special():
    def __init__(self, n, Lx, Ly, delta, beta, decay_rate, eta, eta_special,
                  beta_special, delta_special, tmax, n_special, t_transient, vision = 3):  
        
        #BASICS 
        ###########################
        self.n = n  # number of ants
        self.n_special = n_special # number of special ants
        self.num = self.n + self.n_special
        self.eta_special = eta_special
        self.Lx = Lx 
        self.Ly = Ly
        self.LL = Lx - 1
        self.V = Lx*Ly

        self.positions = np.array([[random.randint(0, self.Lx - 1),
                                     random.randint(0, self.Ly - 1)] for _ in range(self.num)])
        self.pheromones = np.zeros((self.Lx, self.Ly))

        self.directions = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], 
                                    [-1, 0], [-1, -1], [0, -1], [1, -1]]) 
        
        weights = np.round( np.arccos(np.dot(norm_vec(self.directions), norm_vec(self.directions).T)) * 180 / np.pi)
        dict_weights = {0: 1, 45: 1/2, 90: 1/4, 135: 1/12, 180: 1/50}
        key_to_val = lambda key: dict_weights[key] if key in dict_weights else key
        trans_probs = np.array( list(map(lambda row: list(map(key_to_val, row)), weights)) )

        self.transition_probs = trans_probs
        self.head = np.random.randint(0,8,self.num)   

        self.eta = eta # amount of pheromones left by the ants
        self.delta_real = delta
        self.beta_real = beta

        self.delta = np.ones(self.num)*self.delta_real # parameter for the weight function
        self.beta = np.ones(self.num)*self.beta_real # parameter for the weight function
        self.delta_special = delta_special
        self.beta_special = beta_special
        self.decay_rate = decay_rate # decay rate of the pheromones
        self.amount = np.ones(self.num)*eta 
        self.amount[n:] = 0
        
        self.done = False
        self.t = 0
        self.t_transient = t_transient
        self.tmax = tmax
        #########################################
        
        self.s0 = np.sum(self.pheromones)/self.V
        self.rho = self.compute_density()
        self.OP = 0
        self.vision = vision
        self.best_choice_count = 0

        # deque to store the last 20 values of OP
        #self.op_hist = deque(maxlen=20)
        
        #for i in range(20):
        #    self.op_hist.append(0)

    # ANTS SIMULATION
    ##############################################
    # update the pheromones on the grid when the ants pass by
    def pheromone_update(self):

        # vectorized version of the update
        self.pheromones += np.bincount(self.positions[:, 0]*self.Ly + self.positions[:, 1],
                                        minlength=self.Lx*self.Ly, 
                                        weights=self.amount).reshape(self.Lx, self.Ly)
        
    
    # exponential decay of the pheromones
    def pheromone_decay(self):
        self.pheromones *= np.exp(-self.decay_rate)
    
    
    def new_head(self):
        # Vectorized indices calculation for new positions
        indices = (self.positions[:, None, :] + self.directions) % np.array([self.Lx, self.Ly])
        
        # Extract pheromone values at new positions
        pheromones = self.pheromones[indices[:, :, 0], indices[:, :, 1]]

        # Compute transition weights
        w = np.array([self.transition_probs[h] for h in self.head])

        # Compute probabilities for all ants
        p_ij = np.zeros((self.num, 8))
       
        for i in range(self.num): 
            p_ij[i] = phr_weight(pheromones[i], self.delta[i], self.beta[i]) * w[i]
            p_ij[i] /= p_ij[i].sum()  # Normalize probabilities'''
                
        # Choose new direction using vectorized random choice
        rand_indices = np.array([np.random.choice(8, p=p_ij[i]) for i in range(self.num)])
        
        # Update head directions
        return rand_indices


    def update(self): 

        # update the direction of the ants
        #self.head = compute_new_head(self.positions, self.directions, self.head, self.pheromones, 
        #    self.transition_probs, self.delta, self.beta, self.Lx, self.Ly
        #)  

        self.head = self.new_head()
        # self.new_head()
        # update the position of the ant with p.b.c.
        self.positions = (self.positions + self.directions[self.head]) % np.array([self.Lx, self.Ly])
        self.pheromone_update()
        self.pheromone_decay()
        self.t += 1
        
        return
    
    def compute_density(self):

        rho = np.bincount(np.array(self.positions[:self.n,0]*self.Ly + self.positions[:self.n,1]), minlength=self.Lx*self.Ly)
        self.rho = rho.reshape(self.Lx, self.Ly)

        return self.rho

    def compute_density_spec(self):

        rho = np.bincount(np.array(self.positions[self.n:,0]*self.Ly + self.positions[self.n:,1]), minlength=self.Lx*self.Ly)
        self.rho_spec = rho.reshape(self.Lx, self.Ly)

        return self.rho_spec
    
    def compute_density_all(self):
        rho = np.bincount(np.array(self.positions[:,0]*self.Ly + self.positions[:,1]), minlength=self.Lx*self.Ly)
        self.rho = rho.reshape(self.Lx, self.Ly)

        return self.rho
    
    def order_parameter_s(self):

        self.avg_pher()
        plus_id = self.pheromones>self.s0
        minus_id = self.pheromones<=self.s0
        n_plus = np.sum(plus_id)
        n_minus = np.sum(minus_id)
        if n_plus == 0 or n_minus == 0:
            return 0
        else:
            s_plus = np.sum(self.pheromones[plus_id])
            s_minus = np.sum(self.pheromones[minus_id])
            return (s_plus/n_plus - s_minus/n_minus)/(self.s0*self.V)*n_plus
    
    def order_parameter_p(self):

        self.avg_pher()
        plus_id = self.pheromones>self.s0
        minus_id = self.pheromones<=self.s0
        n_plus = np.sum(plus_id)
        n_minus = np.sum(minus_id)

        self.compute_density()

        if n_plus == 0 or n_minus == 0:
            return 0
        else:
            mu_plus = np.sum(self.rho[plus_id])
            mu_minus = np.sum(self.rho[minus_id])
            return (mu_plus - mu_minus)/(mu_plus + mu_minus)

    def transient(self):
        
        for _ in range(self.t_transient):
            self.update()
            self.avg_pher()
            self.OP = self.order_parameter_p()
            #self.op_hist.append(self.OP)
        
        self.amount[self.n:] = self.eta_special
        self.positions[self.n:] = np.array([[random.randint(0, self.Lx - 1),
                                     random.randint(0, self.Ly - 1)] for _ in range(self.n_special)])


    def avg_pher(self):

        self.s0 = np.mean(self.pheromones)

        return self.s0


    def reset_objects(self):
        # Reset the ants' positions
        self.positions = np.array([[random.randint(0, self.Lx - 1),
                                    random.randint(0, self.Ly - 1)] for _ in range(self.num)])

        # Reset the pheromones grid
        self.pheromones = np.zeros((self.Lx, self.Ly))

        # Reset the head directions
        self.head = np.random.randint(0,8,self.num)

        # Reset the pheromone amounts
        self.amount = np.ones(self.num) * self.eta
        self.amount[self.n:] = 0

        # Reset the delta and beta parameters
        self.delta = np.ones(self.num)*self.delta_real
        self.beta = np.ones(self.num)*self.beta_real
        self.delta[self.n:] = self.delta_special
        self.beta[self.n:] = self.beta_special

        # Reset the state collections
        self.op_hist = deque(maxlen=20)
        for _ in range(20):
            self.op_hist.append(0)

        # Reset the time counter and other state variables
        self.t = 0
        self.s0 = np.sum(self.pheromones) / self.V
        self.OP = 0
        self.rho = np.zeros((self.Lx, self.Ly))
        self.reward = 0
        self.done = False


    def meshgrid(self, x, y):

        # Compute wrapped indices efficiently using NumPy
        x_offsets = (x + np.arange(-(self.vision//2), (self.vision//2)+1)) % self.Lx
        y_offsets = (y + np.arange(-(self.vision//2), (self.vision//2)+1)) % self.Ly

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    
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
        x_offsets = (x + np.arange(-(self.vision//2), (self.vision//2)+1)) % self.Lx
        y_offsets = (y + np.arange(-(self.vision//2), (self.vision//2)+1)) % self.Ly

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    
    def get_state(self):
        
        states = []
        self.compute_density_spec()
        self.compute_density()

        for i in range(self.n_special):

            j = self.n+i

            xx, yy = self.meshgrid(self.positions[j, 0], self.positions[j, 1])
            state1 = self.pheromones[xx, yy].reshape(1,self.vision,self.vision).copy()
            state2 = self.rho[xx, yy].reshape(1,self.vision,self.vision).copy()
            state3 = self.rho_spec[xx, yy].reshape(1,self.vision,self.vision).copy()
            state4 = np.zeros((1, self.vision, self.vision))
            h1 = self.directions[self.head[j]][0]
            h2 = self.directions[self.head[j]][1]
            state4[0, self.vision//2+h1, self.vision//2+h2] = 1
            s12 = np.concatenate((state1, state2), axis=0)
            s123 = np.concatenate((s12, state3), axis=0)
            s = np.concatenate((s123, state4), axis=0)
            states.append(s)

        return np.array(states)
    
    
    def component_map_nodes_degree(self, G):
        component_map = {}

        # initialize the component map with default values for the whole lattice of dimensions Lx and Ly
        component_map = {(x, y): [0, 0] for x in range(self.Lx) for y in range(self.Ly)}

        for component in nx.connected_components(G):
            component_nodes = len(component)
            size = component_nodes
            if component_nodes < 2:
                continue

            for node in component:
                degree = G.degree(node)
                component_map[node] = [size, degree]

        return component_map

    
    '''def get_reward(self, new_state, ind):
        
        ns_pher = new_state[ind,0]
        ns0 = np.mean(new_state[ind,0])

        plus_id = ns_pher > ns0
        minus_id = ns_pher <= ns0
        n_plus = np.sum(plus_id)
        n_minus = np.sum(minus_id)

        c = self.LL//2
        grid = plus_id.copy()
        grid[c, c] = 1

        G_t1 = self.create_graph_from_grid_local(grid)

        n_nodes = G_t1.number_of_nodes()
        # compute the number of edges
        n_edges = G_t1.number_of_edges()

        if n_nodes == 1:
            edge_pen = -1
        else:
            edge_pen = -n_edges/(n_nodes-1)

        pop = self.order_parameter_p()
        pod = self.order_parameter_s()

        # print(f"Reward components: edge_pen={edge_pen:.3f}, pop={pop:.3f}, pod={pod:.3f}, n_nodes={n_nodes}, n_edges={n_edges}")
        
        return edge_pen + (pop*min(n_nodes,self.Lx)/self.Lx+pod)/2   '''  


    def get_reward(self):

        # new_state = self.get_global_state()\
        ns_pher = self.pheromones.copy()
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

            pop = self.order_parameter_s()
            pod = self.order_parameter_p()

            len_rew = min(n_nodes,self.Lx)/self.Lx
            rew = edge_pen + (pop*len_rew+pod)/2
        
            metrics_for_components.append([n_edges, n_nodes, rew])

        return np.array(metrics_for_components)

        '''if n_plus == 0 or n_minus == 0:
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

            len_rew = min(n_nodes,self.Lx)/self.Lx
            rew = edge_pen + (pop*len_rew+pod)/2
            rewards_for_components.append([rew, len_rew, edge_pen])
        
        # reward = np.sum(rewards_for_components) if rewards_for_components else 0
        return rewards_for_components, pop, pod'''  
    
    

    def get_global_state(self):
        
        states = []
        self.compute_density_spec()
        self.compute_density()

        for i in range(self.n_special):

            j = self.n+i

            xx, yy = self.global_meshgrid(self.positions[j, 0], self.positions[j, 1])
            state1 = self.pheromones[xx, yy].reshape(1,self.LL,self.LL).copy()
            state2 = self.rho[xx, yy].reshape(1,self.LL,self.LL).copy()
            state3 = self.rho_spec[xx, yy].reshape(1,self.LL,self.LL).copy()
            state4 = np.zeros((1, self.LL, self.LL))
            h1 = self.directions[self.head[j]][0]
            h2 = self.directions[self.head[j]][1]
            state4[0, self.LL//2+h1, self.LL//2+h2] = 1
            s12 = np.concatenate((state1, state2), axis=0)
            s123 = np.concatenate((s12, state3), axis=0)
            s = np.concatenate((s123, state4), axis=0)
            states.append(s)

        return np.array(states)
    
    def global_meshgrid(self, x, y):

        # Compute wrapped indices efficiently using NumPy
        x_offsets = (x + np.arange(-(self.LL//2), (self.LL//2)+1)) % self.LL
        y_offsets = (y + np.arange(-(self.LL//2), (self.LL//2)+1)) % self.LL

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    