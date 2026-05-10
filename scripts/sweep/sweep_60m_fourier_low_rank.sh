#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=mhong

module load gcc/11.3.0
module load conda
module load cuda/12.0
module list
source activate eff

wandb login a98ea5d2a400c0eef0e40399bbb22aa66cf4faa3
export 'WANDB_ENTITY=weiquan0128-university-of-minnesota'

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

nvidia-smi
which python3
echo "Job is starting on $(hostname)"

export LD_LIBRARY_PATH=$HOME/.conda/envs/eff/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HUB_ETAG_TIMEOUT=500
cd /home/mhong/wei00355/code/parameter_efficient_pretraining || exit


# # Run the wandb sweep command and capture both stdout and stderr
# SWEEP_OUTPUT=$(wandb sweep --project sweep-fourier-low-rank scripts/sweep/sweep_60m_fourier_low_rank.yaml 2>&1)

# echo "Sweep OUTPUT: $SWEEP_OUTPUT"

# # Extract the sweep ID using awk
# SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk '/Creating sweep with ID:/ {print $NF}')

# # You can now use SWEEP_ID in subsequent commands
# echo "Sweep ID: $SWEEP_ID"

# # Example: Running an agent with the sweep ID
# wandb agent weiquan0128-university-of-minnesota/sweep-fourier-low-rank/$SWEEP_ID

# wandb sweep --project sweep-fourier-low-rank sweep_60m_fourier_low_rank.yaml
wandb agent --count 1 weiquan0128-university-of-minnesota/sweep-fourier-low-rank/4ny8s0wx