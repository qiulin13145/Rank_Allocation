#!/bin/bash

#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=test-h100
#SBATCH --requeue
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=msigpu

module load cuda
module list
export CONDA_ENVS_PATH="/home/mhong/li003755/.conda/envs"
source activate llm

wandb login b6a06c7d2dc92c75b1bbf5055b3e474b4e943ef0
export WANDB_ENTITY="mhong-university-of-minnesota"
export WANDB_PROJECT="sltrain_qwen_test"

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

#nvidia-smi
which python3
echo "Job is starting on $(hostname)"

export export LD_LIBRARY_PATH=$HOME/.conda/envs/llm/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HUB_ETAG_TIMEOUT=500
cd /home/mhong/li003755/parameter_efficient_pretraining || exit

# LLaMA-1B
torchrun --standalone --nproc_per_node 1 torchrun_main_DDP_no_sync.py \
    --keep_only_last_model \
    --model_name qwen2_0.5b_restart_sltrain_cl200 \
    --model_config configs/qwen2_0.5b.json \
    --use_hf_model \
    --lr 0.005 \
    --peft_model sltrain \
    --optimizer adamW \
    --rank 512 \
    --sp_ratio 0.1 \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 140000 \
    --warmup_steps 14000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 16 \
    --save_every 5000 \
    --cycle_length 200 \
    --weight_decay 0.0 \
    --scheduler cosine_quick_recovery\
    --restart_warmup_steps 10 \
    --seed 42  \

exit
