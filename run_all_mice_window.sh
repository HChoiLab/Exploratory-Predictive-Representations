#!/usr/bin/env bash
set -euo pipefail

# BOUTS=("A1a" "A1b" "B1" "B2" "B3" "B4" "B5" "B6" "B7" "C1" "C3" "C6" "C7" "C8" "C9" "D3" "D4" "D5" "D7" "D7a" "D7b" "D8" "D9" "D9a" "D9b" "F2a" "F2b")

BOUTS=("B5" "C8")
Z_DIMS=(16)
HIDDEN_DIMS=(64)

BETAS=("1e-3")
ONLINE_LR_PRIOR="1e-3"
ONLINE_LR_POSTERIOR="1e-3"
ONLINE_LR_OTHER="1e-3"
SPARSE_BETAS=("0.1")
WINDOWS=(20)

for window_size in "${WINDOWS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for sparse_beta in "${SPARSE_BETAS[@]}"; do
      for bout in "${BOUTS[@]}"; do
        for z in "${Z_DIMS[@]}"; do
          for h in "${HIDDEN_DIMS[@]}"; do
            bouts_dir="data/${bout}-tf"
            out_dir="runs_${window_size}_window/water_sparsity_${sparse_beta}_hdim_${h}_zdim_${z}_beta${beta}_lr${ONLINE_LR_PRIOR}_${ONLINE_LR_POSTERIOR}_${ONLINE_LR_OTHER}/${bout}"
            summary_csv="${out_dir}/summary.csv"

            mkdir -p "${out_dir}"
            mkdir -p "${out_dir}/ckpts"

            echo "Running: window_size=${window_size} bout=${bout} z=${z} h=${h} beta=${beta} online_lr=${ONLINE_LR_PRIOR}/${ONLINE_LR_POSTERIOR}/${ONLINE_LR_OTHER}"
            python -m active_sensing.train_model_windows \
              --bouts-dir "${bouts_dir}" \
              --beta "${beta}" \
              --online-lr-prior "${ONLINE_LR_PRIOR}" \
              --online-lr-posterior "${ONLINE_LR_POSTERIOR}" \
              --online-lr-other "${ONLINE_LR_OTHER}" \
              --metrics-out-dir "${out_dir}" \
              --summary-csv "${summary_csv}" \
              --z-dim "${z}" \
              --waterport-node 116 \
              --sparse_beta "${sparse_beta}" \
              --reward_beta 1  \
              --hidden-dim "${h}" \
              --train-window "${window_size}" \
              --online-ckpt-dir "${out_dir}/ckpts"
          done
        done
      done
    done
  done
done
