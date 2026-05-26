# Exploratory Predictive Representations

This repository contains code for training and analyzing predictive representations in a binary-tree maze. The project studies how exploratory and reward-driven behavioral experience shapes the geometry of latent representations learned by a predictive-coding model.

The model is trained on either self-generated agent trajectories or mouse trajectories from a labyrinth task. It predicts the next maze state and reward probability, and its latent space is used to analyze how exploration-exploitation balance affects representational geometry.

## Repository structure

- `active_sensing/core/` contains the predictive-coding perception model and data loading code.
- `active_sensing/train_model_windows.py` trains the perception model on mouse trajectory windows.
- `active_sensing/action_selection_value_map.py` trains the active agent with value-map-based reward-driven action selection.
- `active_sensing/*.ipynb` contains notebooks used for figure generation and additional analyses.
- `active_sensing/utils/` contains helper functions.
- `outdata/` contains mouse trajectory data from Rosenberg et al. [1].
- `MM_Maze_Utils.py`, `MM_Plot_Utils.py`, and `MM_Traj_Utils.py` contain maze and trajectory utilities adapted from Rosenberg et al. [1].
- `run_all_mice_window.sh` trains the model on mouse trajectory.
- `run_agent_array_rew_driven.slurm` runs active-agent training.

## Reference

[1] Matthew Rosenberg, Tony Zhang, Pietro Perona, and Markus Meister.  
*Mice in a labyrinth show rapid learning, sudden insight, and efficient exploration.*  
eLife, 10:e66175, July 2021.  
doi: [10.7554/eLife.66175](https://doi.org/10.7554/eLife.66175)
