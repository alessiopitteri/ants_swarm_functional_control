import numpy as np
from collections import deque
from utils import *
import random
import pygame
import multiprocessing
from multiprocessing import Pool

# weight function
def phr_weight(sigma, delta, beta):
    return (1 + sigma/(1+delta*sigma))**beta

def compute_probability(args):
    i, pheromones, delta, beta, w = args
    p_i = phr_weight(pheromones[i], delta[i], beta[i]) * w[i]
    p_i /= p_i.sum()  # Normalize probabilities
    return p_i

class Ants():
    def __init__(self, n, Lx, Ly, delta_list, beta_list, decay_rate, eta, eta_special,
                  tmax, n_special, t_transient, vision = 3):  
        
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

        self.delta_list = delta_list
        self.beta_list = beta_list

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
        # self.head = np.array( [random.choice(self.directions) for _ in range(self.num)] )   

        self.eta = eta # amount of pheromones left by the ants
        self.delta_real = np.random.choice(self.delta_list) # parameter for the weight function
        self.beta_real = np.random.choice(self.beta_list) # parameter for the weight function

        self.delta = np.ones(self.num)*self.delta_real # parameter for the weight function
        self.beta = np.ones(self.num)*self.beta_real # parameter for the weight function
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
        self.reward = 0
        self.OP = 0
        self.vision = vision

        

        
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
        p_ij = np.zeros((self.n, 8))
       
        for i in range(self.n): 
            p_ij[i] = phr_weight(pheromones[i], self.delta[i], self.beta[i]) * w[i]
            p_ij[i] /= p_ij[i].sum()  # Normalize probabilities'''
                
        # Choose new direction using vectorized random choice
        rand_indices = np.array([np.random.choice(8, p=p_ij[i]) for i in range(self.n)])
        
        # Update head directions
        return rand_indices


    def update(self): 

        # update the direction of the ants
        #self.head = compute_new_head(self.positions, self.directions, self.head, self.pheromones, 
        #    self.transition_probs, self.delta, self.beta, self.Lx, self.Ly
        #)
        self.head[:self.n] = self.new_head()
        # self.new_head()
        # update the position of the ant with p.b.c.
        self.positions = (self.positions + self.directions[self.head]) % np.array([self.Lx, self.Ly])
        # pheromon update
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
        self.delta_real = np.random.choice(self.delta_list)
        self.beta_real = np.random.choice(self.beta_list)
        self.delta = np.ones(self.num) * self.delta_real
        self.beta = np.ones(self.num) * self.beta_real

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
    
    def draw(self, screen, screen_scale, arrow_size):


        for i in range(self.num):
            
            if i < self.n:

                head = self.directions[self.head[i]]
                perp_head = perpendicular(head)

                pygame.draw.polygon(screen, (0, 0, 0), 
                            (  ((self.positions[i,0]+head[0]*arrow_size)*screen_scale,
                                (self.positions[i,1]+head[1]*arrow_size)*screen_scale),
                            ((self.positions[i,0]-(head[0]+perp_head[0]/2)*arrow_size)*screen_scale,
                                (self.positions[i,1]-(head[1]+perp_head[1]/2)*arrow_size)*screen_scale),
                            ((self.positions[i,0]-(head[0]-perp_head[0]/2)*arrow_size)*screen_scale,
                                (self.positions[i,1]-(head[1]-perp_head[1]/2)*arrow_size)*screen_scale)) )
                
            else:
                pygame.draw.circle(screen, (255, 0, 0), 
                                    (int(self.positions[i,0]*screen_scale), int(self.positions[i,1]*screen_scale)), 10)
