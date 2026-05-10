#! /bin/bash

#S -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1_dev
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M andi.han@riken.jp

# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

python -m wandb login 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
export WANDB_PROJECT="rank_allocation_lora"

torchrun --standalone --nproc_per_node 1 torchrun_main_DDP.py \
    --model_name rank_allocation_lora_60m_42_0.003_500 \
    --model_config configs/llama_60m.json \
    --lr 0.003 \
    --peft_model rank_allocation_lora \
    --optimizer adamW \
    --rank 128 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 32 \
    --save_every 50000 \
    --weight_decay 0.0 \
    --cycle_length 500 \
    --scheduler cosine_quick_recovery \
    --restart_warmup_steps 10 \
    --rank_allocation_delta 8 \
    --rank_allocation_top_k 4 \
    --rank_allocation_min_ratio 0.125 \
    --rank_allocation_max_ratio 0.5 \
    --rank_allocation_hysteresis 0.1 \
    --rank_allocation_probe_rank 1 \
    --rank_allocation_probe_sigma 1e-3 \
    --rank_allocation_credit_sample_interval 20 \
    --rank_allocation_credit_beta 0.95 \
    --rank_allocation_probe_beta 0.9 \
    --dataset_path /data/datasets/c4/en \
    --seed 42
