import numpy as np
from abc import ABC, abstractmethod

class RL(ABC):

    def __init__(self, env):

        self.env = env    
    
    # Initialize the neural networks and optimizers
    @abstractmethod
    def setup_networks(self):
        pass
    
    # Training the agent(s)
    @abstractmethod
    def training_episode(self):
        pass
    
    # Testing the agent(s)
    @abstractmethod
    def test_episode(self):
        pass
    
    # Function to infer the action to take given a state
    # should return an array of actions of dimension (n_agents, action_dim)
    @abstractmethod
    def policy(self, state):
        pass
    
    # Function to obtain the state of the environment
    # should return an array of states of dimension (n_agents, state_dim)
    @abstractmethod
    def get_state(self):
        pass 
    
    # Fuction to execute the action in the environment
    # should return the new state, reward and done flag
    @abstractmethod
    def execute_action(self, action):
        pass

