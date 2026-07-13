import sys
import os
import scipy
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ddqn.DDQN import *

class DDQN_ants_trivia(DDQN):

    def __init__(self, env, replay_buffer,
                 gamma, action_bound, lerning_rate, tau, batch_size, epsilon_decay, weight_decay, config_net_file):
        super().__init__(env, replay_buffer,
                 gamma, action_bound, lerning_rate, tau, batch_size, epsilon_decay, weight_decay, config_net_file)
        
        self.env.gauss = self.gaussuian_filter(self.env.Lx, 0.9, 0)

    def meshgrid(self, x, y):

        # Compute wrapped indices efficiently using NumPy
        x_offsets = (x + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Lx
        y_offsets = (y + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Ly

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    
    def gaussuian_filter(self, kernel_size, sigma=1, muu=0):
 
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

    
    def get_state(self, i):

        # self.env.compute_density()
        xx, yy = self.meshgrid(self.env.positions[self.env.n+i, 0], self.env.positions[self.env.n+i, 1])
        state1 = self.env.gauss[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        #state2 = self.env.rho[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        #return np.concatenate((state1, state2), axis=0)
        return state1

    def execute_action(self, actions):
        
        # p = actions / np.sum(actions, axis=1, keepdims=True)  # Normalize along rows
        #p = scipy.special.softmax(actions, axis=1)
 
        '''dirs = np.array([
            np.random.choice(8, p=row) for row in p
        ])'''
        self.env.head[self.env.n:,:] = self.env.directions[actions]

        self.env.update()

        self.env.op_hist.append(self.env.order_parameter_p())
        
        avg_op = np.mean(np.array(self.env.op_hist))
        self.env.OP = avg_op

        new_states = [self.get_state(i).copy() for i in range(self.env.n_special)]        
        rewards = [self.get_state(i).copy()[:,1,1] for i in range(self.env.n_special)]  

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done  
    

    def policy(self, states, explore=True):
    
        if (random.random() < self.epsilon) and explore:
            return np.random.randint(0, self.replay_buffer.action_dim, size=states.shape[0])
        else:
            state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            with torch.no_grad():
                return self.critic(state_tensor).argmax(axis=1).numpy()
            
    def training_episode(self):

        self.env.reset_objects()
        self.env.transient()

        reward_collection = []
        action_collection = []

        states = [self.get_state(i) for i in range(self.env.n_special)]

        while not self.env.done:

            actions = self.policy(np.array(states))#[:,0]
            new_states, rewards, done = self.execute_action(actions)
            
            reward_collection.append(rewards)
            action_collection.append(actions)

            for i in range(self.env.n_special):
                self.replay_buffer.add_to_buffer(states[i], actions[i], rewards[i], new_states[i], done)
            
            if self.replay_buffer.size >= self.batch_size:

                self.training_step()
            
            states = new_states.copy()
        
        return np.mean(reward_collection, axis=1), np.mean(action_collection, axis=1)
    

    def test_episode(self):

        self.env.reset_objects()
        self.env.transient()
        states = [self.get_state(i) for i in range(self.env.n_special)]

        reward_collection = []
        action_collection = []
        states_collection = [states.copy()]

        done = False

        while not done:
            
            actions = self.policy(states, explore=False)#[:,0]

            new_states, rewards, done = self.execute_action(actions)
            reward_collection.append(rewards)
            action_collection.append(actions.copy())
            states_collection.append(new_states.copy())
            states = new_states.copy()

        return reward_collection, action_collection, states_collection

