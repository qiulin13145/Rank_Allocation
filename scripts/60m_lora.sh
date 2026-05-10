#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=mhong

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
cd /users/3/glent007/workspace/pretrain2/parameter_efficient_pretraining  || exit

export WANDB_PROJECT="SLTrain_Restarts"


torchrun --standalone --nproc_per_node 1 torchrun_main_DDP.py \
    --model_name lora \
    --model_config configs/llama_60m.json \
    --lr 0.0015 \
    --peft_model lora \
    --optimizer adamw \
    --rank 128 \
    --lora_alpha 32 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --restart_every 99999\
    --cycle_length 11000 \
    --scheduler cosine \
    --seed 42  \
    --save_every 50000 \
 