#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=mhong

# module load gcc/11.3.0
# module load conda
# module load cuda/12.0
# module list
# source activate eff

# wandb login a98ea5d2a400c0eef0e40399bbb22aa66cf4faa3
# export 'WANDB_ENTITY=weiquan0128-university-of-minnesota'

# # Benchmark info
# echo "TIMING - Starting jupyter at: $(date)"

# nvidia-smi
# which python3
# echo "Job is starting on $(hostname)"

# export LD_LIBRARY_PATH=$HOME/.conda/envs/eff/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# export HF_HUB_ETAG_TIMEOUT=500
# cd /home/mhong/wei00355/code/parameter_efficient_pretraining || exit


# export http_proxy="http://127.0.0.1:7890"
# export https_proxy="http://127.0.0.1:7890"

wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8
export export WANDB_PROJECT="draft"

torchrun --standalone --nproc_per_node 2 torchrun_main_DDP.py \
    --wandb_project_name eff-training-60m \
    --model_name fourier_low_rank \
    --model_config configs/llama_60m.json \
    --lr 0.003 \
    --peft_model fourier_low_rank \
    --rank 128 \
    --lora_alpha 32 \
    --optimizer adamw \
    --n_freq 10000 \
    --fourier_scale 128 \
    --batch_size 2 \
    --total_batch_size 4 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000
