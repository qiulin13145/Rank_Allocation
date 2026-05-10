#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-4

module load gcc/11.3.0
module load conda
module load cuda/12.0
module list
source activate pretrain

wandb login b8f38344ec7231ee89baa74ef7209dd5a43df6b2
export 'WANDB_ENTITY=mhong-university-of-minnesota'

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

nvidia-smi
which python3
echo "Job is starting on $(hostname)"

export LD_LIBRARY_PATH=$HOME/.conda/envs/eff/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HUB_ETAG_TIMEOUT=500
cd /users/3/glent007/workspace/pretrain/parameter_efficient_pretraining || exit

# wandb agent --count 12 mhong-university-of-minnesota/parameter_efficient_pretraining-scripts_sweep/??? #$SWEEP_ID

wandb agent mhong-university-of-minnesota/parameter_efficient_pretraining-scripts_sweep/??? #$SWEEP_ID


#galore: tvxcshk1

#low-rank: dkg136py

#low-rank restarts: o3enqry5

#sltrain: n1jjo36w


