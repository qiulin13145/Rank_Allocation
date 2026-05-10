#!/bin/bash

#SBATCH --time=5:00:00
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

wandb login a7334a5d5a86151953b81d59dda0c0e661e41ffd
export 'WANDB_ENTITY=athglentis-university-of-minnesota'

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

#nvidia-smi
which python3
echo "Job is starting on $(hostname)"

export export LD_LIBRARY_PATH=$HOME/.conda/envs/eff/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HUB_ETAG_TIMEOUT=500
cd /users/3/glent007/workspace/pretrain/parameter_efficient_pretraining || exit
export WANDB_PROJECT="SPAM"

 
torchrun --standalone --nproc_per_node 1 torchrun_main_DDP.py \
    --hf_dataset \
    --model_config configs/llama_60m.json \
    --model_name full-rank_sltrain-stable_spam_5e-3_rank_128 \
    --lr 5e-3 \
    --peft_model sltrain \
    --rank 128 \
    --sp_ratio 0.1 \
    --lora_alpha 16 \
    --optimizer stable_spam_adamw \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 10000 \
    --seed 42  \
    --scheduler cosine \
    --cycle_length 10000 \
    --gamma1 0.85 \
    --gamma2 0.99999 \
    --gamma3 0.999 \
    --total_T 10000 \
    --eta 0.5 \
    --density 1.0 \
    --update_gap 1000
 
#     --no_slice \ 