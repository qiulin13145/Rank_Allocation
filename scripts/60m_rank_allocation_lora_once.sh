#! /bin/bash

#S -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1_dev
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M andi.han@riken.jp

# Delayed Static Rank Allocation:
# 0-4000 steps collect score with uniform rank, allocate once at step 4000,
# then keep rank fixed and continue restart_lora-style refactorization.

python -m wandb login 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
export WANDB_PROJECT="rank_allocation_lora"

torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
    --model_name once_16_64 \
    --model_config configs/llama_60m.json \
    --lr 0.003 \
    --peft_model rank_allocation_lora \
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
    --scheduler cosine_quick_recovery \
    --restart_warmup_steps 10 \
    --rank_allocation_delta 16 \
    --rank_allocation_top_k 64 \
    --rank_allocation_min_ratio 0.1 \
    --rank_allocation_max_ratio 0.5 \
    --rank_allocation_hysteresis 0.0 \
    --rank_allocation_probe_rank 8 \
    --rank_allocation_probe_sigma 1e-3 \
    --rank_allocation_credit_sample_interval 5 \
    --rank_allocation_credit_beta 0.95 \
    --rank_allocation_probe_beta 0.9 \
    --rank_allocation_start_step 4000 \
    --rank_allocation_interval 500 \
    --rank_allocation_once \
    --rank_allocation_tail_threshold 0.3 \
    --rank_allocation_extra_init_std 1e-4 \
    --rank_allocation_report_dir log/rank_allocation_reports \
    --dataset_path /data/datasets/c4/en \
    --seed 42
