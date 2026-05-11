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

torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
    --model_name rank_allocation_lora_60m_conservative_42_0.003_500 \
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
    --rank_allocation_delta 8 \
    --rank_allocation_top_k 2 \
    --rank_allocation_min_ratio 0.1875 \
    --rank_allocation_max_ratio 0.3125 \
    --rank_allocation_hysteresis 0.0 \
    --rank_allocation_probe_rank 8 \
    --rank_allocation_probe_sigma 1e-3 \
    --rank_allocation_credit_sample_interval 20 \
    --rank_allocation_credit_beta 0.95 \
    --rank_allocation_probe_beta 0.9 \
    --rank_allocation_start_step 4000 \
    --rank_allocation_interval 2000 \
    --rank_allocation_tail_threshold 5e-3 \
    --rank_allocation_extra_init_std 1e-4 \
    --rank_allocation_report_dir log/rank_allocation_reports \
    --dataset_path /data/datasets/c4/en \
    --seed 42
