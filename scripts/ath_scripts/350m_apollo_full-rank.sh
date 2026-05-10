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

torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --keep_only_last_model \
    --model_name 350m_apollo_lr_1e-2_r1024 \
    --model_config configs/llama_350m.json \
    --lr 0.01 \
    --rank 1024 \
    --peft_model apollo \
    --optimizer apollo_adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 5000 \
    --cycle_length 60000 \
    --weight_decay 0.0 \
    --scheduler cosine \
    --update_proj_gap 200 \
    --seed 42  \
    --apollo_scale 1.0 \
    --scale_type channel \
    --proj random \

# !!!     --no_slice \
# /usr/bin/shutdown
