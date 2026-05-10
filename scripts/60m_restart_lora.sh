#! /bin/bash

#S -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1_dev
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M andi.han@riken.jp

#source ./env_setup.sh


python -m wandb login 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
export WANDB_PROJECT="rank_allocation_lora"

# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
    --model_name restart_lora_42_0.003_500 \
    --model_config configs/llama_60m.json \
    --lr 0.003 \
    --peft_model restart_lora \
    --optimizer adamW \
    --rank 128 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 32 \
    --save_every 50000 \
    --weight_decay 0.0 \
    --cycle_length 500 \
    --scheduler cosine_quick_recovery\
    --restart_warmup_steps 10 \
    --dataset_path /data/datasets/c4/en \
    --seed 42  \
     
 
## 更小的学习率（防止Loss突然炸），无reset（防止梯度过大，但不知道为什么），重启后学习率热身（防止Loss突然炸）


# /usr/bin/shutdown