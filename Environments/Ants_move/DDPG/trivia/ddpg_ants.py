import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Algorithms.ddpg.DDPG import *

class DDPG_ants(DDPG):

    def __init__(self, env, buffer,
                 gamma, action_lower_bound, action_upper_bound, actor_lr, critic_lr, tau, batch_size, noise, weight_decay,
                   config_net_file):
        super().__init__(env, buffer, 
                 gamma, action_lower_bound, action_upper_bound, actor_lr, critic_lr, tau, batch_size, noise, weight_decay,
                   config_net_file)
        

    def get_state(self, i):

        # self.env.compute_density()
        xx, yy = self.meshgrid(self.env.positions[self.env.n+i, 0], self.env.positions[self.env.n+i, 1])
        state1 = self.env.gauss[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        #state2 = self.env.rho[xx, yy].reshape(1,self.env.vision,self.env.vision).copy()
        #return np.concatenate((state1, state2), axis=0)
        return state1

    def execute_action(self, actions):
                
        self.env.head[self.env.n:,:] = self.env.directions[actions]

        self.env.update()

        # self.env.op_hist.append(self.env.order_parameter_p())
        
        # avg_op = np.mean(np.array(self.env.op_hist))
        self.env.OP = self.env.order_parameter_p()

        new_states = [self.get_state(i).copy() for i in range(self.env.n_special)]        
        rewards = [self.get_state(i).copy()[:,1,1] for i in range(self.env.n_special)]  

        if self.env.t >= self.env.tmax:
            self.env.done = True
        
        return new_states, rewards, self.env.done  

