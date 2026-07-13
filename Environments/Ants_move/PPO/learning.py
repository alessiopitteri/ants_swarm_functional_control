#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

# Add parent directory for ants_move import
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from ants_move import *

# Get and confirm work folder
work_folder = input('work folder: ')  # Give full path or relative to this script
os.chdir(work_folder)

'''# Confirm where we are and what's inside
print("Current directory:", os.getcwd())
print("Contents:", os.listdir())'''

# Add work folder to sys.path
abs_work_folder = os.path.abspath(os.getcwd())
if abs_work_folder not in sys.path:
    sys.path.append(abs_work_folder)

# Try import
from ppo_ants import PPO_ants

# Saving folder input
saving_folder = input('saving folder: ')



# In[2]:


# set seeds 
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)


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



# In[6]:
file_path = saving_folder + '/' + 'parameters.txt'  # Path to the parameters file

parameters = read_parameters(file_path)

for key, value in parameters.items():
    print(f"{key} = {value}")


# In[7]:


env = Ants(n, Lx, Ly, delta, beta, decay_rate, eta, eta_special,
            tmax, n_special, t_transient, vision=state_dim[-1])


# In[8]:
config_net_file = f"{saving_folder}/networks.yaml"
ppo_ants = PPO_ants(env,
                 gamma, action_bound, actor_lr, critic_lr, batch_size, weight_decay, config_net_file,
                 clip_epsilon, entropy_coeff, policy_steps, learning_steps, 
                 state_dim, action_dim, clip_grad,
                   alpha)
        


# In[9]:


ppo_ants.setup_networks()

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


print(ppo_ants.critic)


# In[12]:


cw = ppo_ants.compute_critic_weights()
indx_w_critic = []

for i in range(len(cw)):
    try:
        cw[i].numpy().shape[1]
        indx_w_critic.append(i)
    except:
        pass

aw = ppo_ants.compute_actor_weights()
indx_w_actor = []

for i in range(len(aw)):
    try:
        aw[i].numpy().shape[1]
        indx_w_actor.append(i)
    except:
        pass


# In[13]:

epi = 0
N_epi = input('Number of episodes: ')
N_epi = int(N_epi)


# In[16]:

best_score = -np.inf

while epi < N_epi:
        
    score, act = ppo_ants.training_episode()

    if epi %1 == 0:

        avg_score = np.mean(score)
        avg_a = np.mean(act)
        final_op = ppo_ants.env.order_parameter_p().copy()
        final_op_s = ppo_ants.env.order_parameter_s().copy()
        
        if epi %10 == 0:
                
            score_test, act_test = ppo_ants.test_episode()
                
            with open(saving_folder + '/results/reward.txt', 'a') as file:
                for item in score:
                    file.write(f"{item}\n")

            with open(saving_folder + '/results_test/reward.txt', 'a') as file:
                for item in score_test:
                    file.write(f"{np.mean(item)}\n")

        cw = ppo_ants.compute_critic_weights()
        aw = ppo_ants.compute_actor_weights()

        with open(saving_folder + '/results/critic_weights.txt', 'a') as f:
            f.write(str([cw[i].numpy().mean() for i in indx_w_critic]) + '\n')
        
        with open(saving_folder + '/results/actor_weights.txt', 'a') as f:
            f.write(str([aw[i].numpy().mean() for i in indx_w_actor]) + '\n')

        ppo_ants.save_weights(saving_folder + '/weights')

    
    with open(saving_folder + '/results/results.txt', 'a') as f:
                    
        f.write(str(epi) + ' '  + str(avg_score) + ' ' + str(avg_a) + ' ' + str(final_op) + ' ' + str(final_op_s) + '\n')

    epi += 1 
    print('epi: ', epi, 'score: ', np.mean(score), end='\r')

    if np.mean(score) > best_score:
        best_score = np.mean(score)
        ppo_ants.save_weights(saving_folder + '/weights/best_weights')
        # print('New best score: ', best_score, end='\r')

