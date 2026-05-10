#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-4

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

# export export LD_LIBRARY_PATH=$HOME/.conda/envs/eff/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# export HF_HUB_ETAG_TIMEOUT=500
# cd /home/mhong/wei00355/code/parameter_efficient_pretraining || exit

# export http_proxy="http://127.0.0.1:7890"
# export https_proxy="http://127.0.0.1:7890"

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

python -m wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8
# export WANDB_PROJECT="3"
# export WANDB_PROJECT="seed52"
export WANDB_PROJECT="opt_project"

torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --model_name sltrain \
    --model_config configs/llama_60m.json \
    --lr 0.003 \
    --peft_model sltrain \
    --optimizer adamw \
    --rank 128 \
    --lora_alpha 32 \
    --sp_ratio 0.03 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000
