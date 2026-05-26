#!/bin/bash
#SBATCH --job-name=agent_grid
#SBATCH --output=logs/agent_grid_%A_%a.out
#SBATCH --error=logs/agent_grid_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-419%20
#SBATCH --account=
#SBATCH --qos=inferno

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


rew_p_lst=("0.015" "0.02" "0.03" "0.035" "0.04" "0.045" "0.055" "0.06" "0.065" "0.07" "0.08" "0.085" "0.09" "0.095" )
n_rew=${#rew_p_lst[@]}

task_id=${SLURM_ARRAY_TASK_ID}
seed=$(( task_id / n_rew ))
rew_idx=$(( task_id % n_rew ))
rew_p="${rew_p_lst[$rew_idx]}"


# mkdir -p "REWARD_DRIVEN_1000"
mkdir -p "REWARD_DRIVEN_1000/REWARD_DRIVEN_1000_runs_slurm_20_window_agent_rew_p_${rew_p}/${seed}"


python -m active_sensing.action_selection_value_map \
    --num-bouts 1000 \
    --horizon 0 \
    --k-info-gain 20 \
    --metrics-out-dir "REWARD_DRIVEN_1000/REWARD_DRIVEN_1000_runs_slurm_20_window_agent_rew_p_${rew_p}/${seed}" \
    --summary-csv "REWARD_DRIVEN_1000/REWARD_DRIVEN_1000_runs_slurm_20_window_agent_rew_p_${rew_p}/${seed}/summary.csv" \
    --train-window 20 \
    --z-dim 16 \
    --hidden-dim 64 \
    --online-lr-prior "1e-3" \
    --online-lr-posterior "1e-3" \
    --online-lr-other "1e-3" \
    --sparse-beta 0.0 \
    --reward-beta 1 \
    --beta "1e-3" \
    --reward-driven-p "${rew_p}" \
    --seed "${seed}" \
    --value-update-every 50 \
    --value-update-window 50 \
    --value-gamma 0.8 \
    --value-iters 100 \
    --value-tau 0.1 