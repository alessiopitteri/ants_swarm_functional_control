import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import Env

# Extract position of the last agent
def meshgrid(x, y, Lx, Ly, vision):

    # Compute wrapped indices efficiently using NumPy
    x_offsets = (x + np.arange(-(vision//2), vision//2+1)) % Lx
    y_offsets = (y + np.arange(-(vision//2), vision//2+1)) % Ly

    # Use NumPy's meshgrid to create index matrices
    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

    return xx, yy

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

class SimpleEnvCNN(Env):

    def __init__(self, max_steps):
        self.directions = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], 
                                    [-1, 0], [-1, -1], [0, -1], [1, -1]]) 
        self.Lx = 25
        self.Ly = 25
        self.n = 10
        self.x = np.random.randint(0,self.Lx, size=self.n)
        self.y = np.random.randint(0,self.Ly, size=self.n)
        self.grid = gaussuian_filter(self.Lx, 0.9, 0) #+ \
                    #gaussuian_filter(self.Lx, 0.5, 1.3)
        
        self.vision = int(3)
        self.state = np.array(
            [self.grid[meshgrid(self.x[i], self.y[i],
                                self.Lx, self.Ly, self.vision)].reshape(1, self.vision, self.vision)
                                  for i in range(self.n)]
                                  )
        self.state.reshape(self.n, 1, self.vision, self.vision)


        self.done = False
        self.max_steps = max_steps
        self.step = 0

    def reset_objects(self):
        self.done = False
        self.x = np.random.randint(0,self.Lx, size=self.n)
        self.y = np.random.randint(0,self.Lx, size=self.n)
        self.state = np.array(
            [self.grid[meshgrid(self.x[i], self.y[i],
                                self.Lx, self.Ly, self.vision)].reshape(1, self.vision, self.vision)
                                  for i in range(self.n)]
                                  )        
        self.state.reshape(self.n, 1, self.vision, self.vision)


        self.step = 0
        return self.state

    def update(self, action):
        
        for i in range(self.n):
            self.x[i] += self.directions[action[i]][0]   
            self.y[i] += self.directions[action[i]][1]    
            self.x[i] = self.x[i] % self.Lx  
            self.y[i] = self.y[i] % self.Ly

        self.step += 1
        if self.step >= self.max_steps:
            self.done = True
            
        self.state = np.array(
            [self.grid[meshgrid(self.x[i], self.y[i],
                                self.Lx, self.Ly, self.vision)].reshape(1, self.vision, self.vision)
                                  for i in range(self.n)]
                                  )        
        self.state.reshape(self.n, 1, self.vision, self.vision)
    
    def transient(self):
        pass
        