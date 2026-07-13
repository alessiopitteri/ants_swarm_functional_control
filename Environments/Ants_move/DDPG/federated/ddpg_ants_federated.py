import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ddpg.DDPG import *

class DDPG_ants_federated(DDPG):

    def __init__(self, env, buffer,
                 gamma, action_bound, actor_lr, critic_lr, tau, batch_size, noise, weight_decay, config_net_file):
        super().__init__(env, buffer, 
                 gamma, action_bound, actor_lr, critic_lr, tau, batch_size, noise, weight_decay, config_net_file)
        

    def meshgrid(self, x, y):

        # Compute wrapped indices efficiently using NumPy
        x_offsets = (x + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Lx
        y_offsets = (y + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Ly

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    
    
    def get_state(self, i):

        self.env.compute_density()
        xx, yy = self.meshgrid(self.env.positions[self.env.n+i, 0], self.env.positions[self.env.n+i, 1])
        state1 = self.env.pheromones[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        state2 = self.env.rho[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        return np.concatenate((state1, state2), axis=0)


    def execute_action(self, actions):
        
        p = actions / np.sum(actions, axis=1, keepdims=True)  # Normalize along rows

        dirs = np.array([
            np.random.choice(8, p=row) for row in p
        ])
        self.env.head[self.env.n:,:] = self.env.directions[dirs]

        self.env.update()

        self.env.op_hist.append(self.env.order_parameter_p())
        
        avg_op = np.mean(np.array(self.env.op_hist))
        self.env.OP = avg_op

        new_states = [self.get_state(i).copy() for i in range(self.env.n_special)]        
        rewards = [avg_op for i in range(self.env.n_special)]  

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done  
    

    def training_episode(self):

        self.env.reset_objects()
        self.env.transient()

        reward_collection = []
        action_collection = []

        states = [self.get_state(i) for i in range(self.env.n_special)]

        while not self.env.done:

            actions = self.policy(states)#[:,0]
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
            
            actions = self.policy(states, test=True)#[:,0]

            new_states, rewards, done = self.execute_action(actions)
            reward_collection.append(rewards)
            action_collection.append(actions.copy())
            states_collection.append(new_states.copy())
            states = new_states.copy()

        return reward_collection, action_collection, states_collection

