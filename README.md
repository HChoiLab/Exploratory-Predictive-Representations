# Running Experiments

Run all commands from the repository root.

### Main experiment (symmetric maze)

```bash
python -m experiments.train_main \
    --num-bouts 1000 --horizon 0 \
    --z-dim 16 --hidden-dim 64 \
    --beta 0.001 --sparse-beta 0.1 --reward-beta 1 \
    --online-lr-prior 0.001 --online-lr-posterior 0.001 --online-lr-other 0.001 \
    --train-window 20 \
    --reward-driven-p 0.02 --seed 42 \
    --metrics-out-dir runs/main \
    --summary-csv runs/main/summary.csv
```

Change `--reward-driven-p` and `--seed` for other conditions.

*For the following experiments, use the same hyperparameters as in the main experiment, changing the script name and the additional arguments shown below.*

### Exploration strategies

```bash
python -m experiments.train_exploration_policies \
    --exploration-policy random \
```

Supported policies: `eig`, `random`, `count`.

### Asymmetric maze

```bash
python -m experiments.train_asymmetric_maze \
    --waterport-node 126 \
```

### History-free model

```bash
python -m experiments.train_history_free 
```

### Mouse data preparation

Mouse behavioral data are available from [Rosenberg et al. (2021)](https://github.com/markusmeister/Rosenberg-2021-Repository).

Download the original recordings and place the `*-tf` files in the `outdata/` directory.

Convert the recordings into per-bout `.npz` files using:

```bash
python -m preprocessing.prepare_mouse_data \
    --tf-dir outdata \
    --out-dir data
```

The script removes consecutive repeated nodes, excludes bouts with fewer than five transitions, and saves node and action sequences for each recording.

### Training on mouse trajectories

To train the predictive model on an individual recording:

```bash
python -m experiments.train_mouse_data \
    --bouts-dir data/B5-tf \
    --z-dim 16 --hidden-dim 64 \
    --beta 0.001 --sparse_beta 0.1 --reward_beta 1 \
    --online-lr-prior 0.001 \
    --online-lr-posterior 0.001 \
    --online-lr-other 0.001 \
    --train-window 20 \
    --metrics-out-dir runs/mouse_B5
```

Replace `B5` with the desired recording identifier.
