#! /bin/bash

#S -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1_dev
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M andi.han@riken.jp

#source ./env_setup.sh

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

python -m wandb login 585b4959ccb98b1ea4d6466883052012b2c9cca8
# export WANDB_PROJECT="seed52"
# export WANDB_PROJECT="draft"
# export WANDB_PROJECT="splora_SVD_restart"
export WANDB_PROJECT="1229_restart"

# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --model_name 残差-restart-5.5K \
    --model_config configs/llama_60m.json \
    --lr 0.002 \
    --peft_model restart \
    --optimizer adamw \
    --rank 64 \
    --sp_ratio 0.03 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 32 \
    --save_every 50000 \
    --weight_decay 0.01 \
    --cycle_length 200 \
    --scheduler cosine\
    --restart_warmup_steps 1 \
    --seed 42  \
     
 
## 更小的学习率（防止Loss突然炸），无reset（防止梯度过大，但不知道为什么），重启后学习率热身（防止Loss突然炸）


# /usr/bin/shutdown