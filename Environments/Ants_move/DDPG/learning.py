#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


# In[2]:


# set seeds 
np.random.seed(10)
torch.manual_seed(1)
random.seed(10)


# In[3]:


def read_parameters(file_path):
    params = {}

    # First pass: read and store values as strings
    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines and comments (if any)
            line = line.strip()
            if line and not line.startswith('#'):
                # Split the line at the equals sign
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()

                # Store the parameter as a string for now
                params[key] = value

    # Second pass: evaluate expressions and convert values
    for key, value in params.items():
        try:
            # Try to evaluate the value as a Python expression using previously defined parameters
            params[key] = eval(value, {}, params)
        except:
            # If it's not an expression, check if it can be converted to float or int
            if '.' in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        
        # Set the variable in the global namespace with the evaluated value
        globals()[key] = params[key]

    return params


# In[4]:


work_folder = input('work folder')
saving_folder = input('saving folder')
os.chdir(work_folder)


# In[5]:


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from ants_move import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
from ddpg.buffer import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "")))

# In[6]:


file_path = saving_folder + '/' + 'parameters.txt'  # Path to the parameters file

parameters = read_parameters(file_path)

for key, value in parameters.items():
    print(f"{key} = {value}")


# In[7]:


env = Ants(n, Lx, Ly, delta, beta, decay_rate, eta, eta_special,
            tmax, n_special, t_transient)

buffer = ReplayBuffer(max_size, state_dim, action_dim)


# In[8]:


from ddpg_ants import *
config_net_file = f"{saving_folder}/networks.yaml"
ddpg_ants = DDPG_ants(env, buffer, gamma, action_bound, actor_lr, critic_lr, 
                                tau, batch_size, noise, weight_decay, config_net_file)


# In[9]:


ddpg_ants.setup_networks()

if not os.path.exists(saving_folder + '/results'):
    os.makedirs(saving_folder + '/results')

if not os.path.exists(saving_folder + '/results_test'):
    os.makedirs(saving_folder + '/results_test')

if not os.path.exists(saving_folder + '/weights'):
    os.makedirs(saving_folder + '/weights')

with open(saving_folder + '/results/results.txt', 'w') as file:
    file.write(f"")

with open(saving_folder + '/results/reward.txt', 'w') as file:
    file.write(f"")
    
'''with open(saving_folder + '/results/op.txt', 'w') as file:
    file.write(f"")'''
    
with open(saving_folder + '/results_test/reward.txt', 'w') as file:
    file.write(f"")

'''with open(saving_folder + '/results_test/op.txt', 'w') as file:
    file.write(f"")'''

with open(saving_folder + '/results/actor_weights.txt', 'w') as file:
    file.write(f"")

with open(saving_folder + '/results/critic_weights.txt', 'w') as file:
    file.write(f"")


# In[10]:


print(ddpg_ants.actor)
print(ddpg_ants.critic)


# In[12]:


aw = ddpg_ants.compute_actor_weights()
indx_w_actor = []

for i in range(len(aw)):
    try:
        aw[i].numpy().shape[1]
        indx_w_actor.append(i)
    except:
        pass

cw = ddpg_ants.compute_critic_weights()
indx_w_critic = []

for i in range(len(cw)):
    try:
        cw[i].numpy().shape[1]
        indx_w_critic.append(i)
    except:
        pass


# In[13]:


epi = 0


# In[14]:


os.getcwd()


# In[29]:


ddpg_ants.policy(np.random.uniform(0,1,(1,1,3,3)), test=False)


# In[31]:

N_epi = int(input('N_epi:'))

while epi < N_epi:
        
    score, act = ddpg_ants.training_episode()

    if epi %1 == 0:
        
        if epi %10 == 0:
                
            score_test, act_test, _ = ddpg_ants.test_episode()
                
            with open(saving_folder + '/results/reward.txt', 'a') as file:
                for item in score:
                    file.write(f"{item}\n")

            with open(saving_folder + '/results_test/reward.txt', 'a') as file:
                for item in score_test:
                    file.write(f"{np.mean(item)}\n")

        aw = ddpg_ants.compute_actor_weights()
        cw = ddpg_ants.compute_critic_weights()

        with open(saving_folder + '/results/actor_weights.txt', 'a') as f:
            f.write(str([aw[i].numpy().sum() for i in indx_w_actor]) + '\n')

        with open(saving_folder + '/results/critic_weights.txt', 'a') as f:
            f.write(str([cw[i].numpy().sum() for i in indx_w_critic]) + '\n')

        ddpg_ants.save_weights(saving_folder + '/weights')

    avg_score = np.mean(score)
    avg_a = np.mean(act)
    
    with open(saving_folder + '/results/results.txt', 'a') as f:
                    
        f.write(str(epi) + ' '  + str(avg_score) + ' ' + str(avg_a) + '\n')

    epi += 1 
    
    print('epi: ', epi, ' score: ', np.sum(score), end='\r')


# In[17]:


ddpg_ants.get_state(0)[:,1,1]


# In[38]:


ddpg_ants.replay_buffer.sample(1)


# In[17]:


s, a, r, ns, d = ddpg_ants.replay_buffer.sample(1000)

r = 0
print(s[r], '\n', ns[r])


# In[14]:


plt.plot(s[:,0].reshape(1000,9), ns[:,0].reshape(1000,9), 'o')

