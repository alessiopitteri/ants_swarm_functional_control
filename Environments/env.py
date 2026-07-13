import numpy as np
from abc import ABC, abstractmethod

class Env(ABC):

    def __init__(self, max_steps):

        self.state = None
        self.max_steps = max_steps
        self.step = 0

    # function to update the state of the environment    
    @abstractmethod
    def update():
        pass

    # function to reset the objects of the environment
    @abstractmethod
    def reset_objects(self):
        pass

    # function to perform a transient step
    @abstractmethod
    def transient(self):
        pass

