#!/bin/bash

#SBATCH --time=8:00:00
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

#nvidia-smi
which python3
echo "Job is starting on $(hostname)"

export export LD_LIBRARY_PATH=$HOME/.conda/envs/eff/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HUB_ETAG_TIMEOUT=500
cd /users/3/glent007/workspace/pretrain/parameter_efficient_pretraining || exit

export WANDB_PROJECT="SLTrain_Restarts"


# LLaMA-1B
torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
    --no_slice \
    --keep_only_last_model \
    --model_name 7b_sltrain_restart \
    --model_config configs/llama_7b.json \
    --lr 0.005 \
    --peft_model sltrain \
    --optimizer adamW \
    --rank 1024 \
    --sp_ratio 0.1 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 500000 \
    --warmup_steps 50000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 16 \
    --save_every 5000 \
    --cycle_length 200 \
    --weight_decay 0.0 \
    --scheduler cosine_quick_recovery \
    --restart_warmup_steps 10 \
    --seed 42  \
    --dataset_path /home/mhong/shared/glent007/c4/en

# /usr/bin/shutdown
