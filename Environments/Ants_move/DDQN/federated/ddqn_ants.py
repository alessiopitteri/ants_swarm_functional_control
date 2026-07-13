import sys
import os
import scipy
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ddqn.DDQN import *

class DDQN_ants(DDQN):

    def __init__(self, env, replay_buffer,
                 gamma, action_bound, lerning_rate, tau, batch_size, epsilon_decay, weight_decay, config_net_file,
                 target_upd_steps, epsilon_min, epsilon_start):
        
        super().__init__(env, replay_buffer,
                 gamma, action_bound, lerning_rate, tau, batch_size, epsilon_decay, weight_decay, config_net_file,
                 target_upd_steps, epsilon_min, epsilon_start)
        

    def meshgrid(self, x, y):

        # Compute wrapped indices efficiently using NumPy
        x_offsets = (x + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Lx
        y_offsets = (y + np.arange(-(self.env.vision//2), (self.env.vision//2)+1)) % self.env.Ly

        # Use NumPy's meshgrid to create index matrices
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing='ij')

        return xx, yy
    
    def get_state(self, i):

        self.env.compute_density_spec()
        xx, yy = self.meshgrid(self.env.positions[self.env.n+i, 0], self.env.positions[self.env.n+i, 1])
        state1 = self.env.pheromones[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        state2 = self.env.rho[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        state3 = self.env.rho_spec[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        s12 = np.concatenate((state1, state2), axis=0)

        return np.concatenate((s12, state3), axis=0)


    def execute_action(self, actions):
        
        self.env.head[self.env.n:,:] = self.env.directions[actions]

        self.env.update()

        #self.env.op_hist.append(self.env.order_parameter_p())
        
        #avg_op = np.mean(np.array(self.env.op_hist))
        #self.env.OP = self.env.order_parameter_s()
        self.env.OP = self.env.order_parameter_p()

        new_states = [self.get_state(i).copy() for i in range(self.env.n_special)]        
        rewards = [self.env.OP for _ in range(self.env.n_special)]  

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
    

    def test_episode(self, explore=False, return_pos=False):

        self.env.reset_objects()
        self.env.transient()
        states = [self.get_state(i) for i in range(self.env.n_special)]

        reward_collection = []
        action_collection = []
        states_collection = [states.copy()]
        pos_collection = [self.env.positions.copy()]

        done = False

        while not done:
            
            actions = self.policy(np.array(states), explore=explore) #[:,0]

            new_states, rewards, done = self.execute_action(actions)
            reward_collection.append(rewards)
            action_collection.append(actions.copy())
            states_collection.append(new_states.copy())
            pos_collection.append(self.env.positions.copy())
            states = new_states.copy()

        if return_pos:
            return reward_collection, action_collection, states_collection, pos_collection

        return reward_collection, action_collection, states_collection

