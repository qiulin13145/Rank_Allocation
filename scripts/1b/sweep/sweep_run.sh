#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100-4

module load gcc/11.3.0
module load conda
module load cuda/11.2
module list
export CONDA_ENVS_PATH="/home/mhong/li003755/.conda/envs"
source activate sltrain

wandb login 3dbaa1026adab988dca53f5bebe8eff91ed0d378
export 'WANDB_ENTITY=jasonljx96'
export 'WANDB_PROJECT=eff_pretrain'

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

nvidia-smi
which python3
echo "Job is starting on $(hostname)"

export LD_LIBRARY_PATH=$HOME/.conda/envs/sltrain/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HUB_ETAG_TIMEOUT=500
cd /home/mhong/li003755/parameter_efficient_pretraining || exit

# wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
# export 'WANDB_PROJECT=sweep'

# wandb sweep scripts/1b/sweep/sweep_1b_fourier_low_rank.yaml

wandb agent --count 1 $@