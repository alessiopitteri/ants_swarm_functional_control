# Ants Swarm Functional Control

Repository for Reinforcement Learning (RL) training, policy evaluation, and benchmark analysis for ant swarm functional control.

---

## Training

Run the training scripts for policy `x = 0, ..., 9`.

### 1. Smart-Setup (Random Parameters)
Training where $\beta$ and $\delta$ are randomly chosen from the parameter sets:
```bash
python Environments/Ants_move_rand/PPO/learning.py
```
> **Output:** Results and network weights are saved in the `edge_spen/lx/` directory.

### 2. Specific Setup
Training on a specific $\beta$-$\delta$ parameter setup:
```bash
python Environments/Ants_move/PPO/learning.py
```

---

## Evaluation & Analysis

* **Phase Diagram & Behavior Visualization:** Run the `test_gridplot.ipynb` notebook to test policies across the entire phase diagram and visualize behavior.
* **Performance Measures:** Compute metrics using the benchmark notebooks:
  * For Random Setup: `Environments/Ants_move_rand/PPO/benchmark_test_*.ipynb`
  * For Specific Setup: `Environments/Ants_move/PPO/benchmark_test_*.ipynb`
