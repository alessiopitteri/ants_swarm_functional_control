# ants_swarm_functional_control
To perform the RL training in the normal smart-setup (beta and delta randomly chosen from the parameters sets) run Environments/Ants_move_rand/PPO/learning.py with policy x = 0,..,9
This will produce as output in the edge_spen/lx folder the results and the network weights.
Then to test the policies in the whole phase diagram and visualize the behavior run the test_gridplot.ipynb notebook.
For the RL training in the specific beta-delta setup and run Environments/Ants_move/PPO/learning.py with policy x = 0,..,9
Measures can be computed in Environments/Ants_move(_rand)/PPO/banchmark_test_*.ipynb