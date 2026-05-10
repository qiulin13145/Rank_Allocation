#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=mhong

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
 
 
torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --no_slice \
    --keep_only_last_model \
    --model_config configs/llama_1b.json \
    --model_name full-rank_spam \
    --lr 2e-4 \
    --peft_model full-rank \
    --optimizer spam_adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 5000 \
    --seed 42  \
    --scheduler cosine \
    --cycle_length 100000 \
    --warmup_epoch 150 \
    --threshold 5000 \
    --grad_accu_steps 20 \
    --density 1.0 \
    --update_gap 500 \
    --dataset_path /home/mhong/shared/glent007/c4/en 
 
