#!/bin/bash

#SBATCH --time=8:00:00
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

wandb login b8f38344ec7231ee89baa74ef7209dd5a43df6b2
export 'WANDB_ENTITY=mhong-university-of-minnesota'

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

#nvidia-smi
which python3
echo "Job is starting on $(hostname)"

export export LD_LIBRARY_PATH=$HOME/.conda/envs/eff/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HUB_ETAG_TIMEOUT=500
cd /code/hongpaul-sandbox/temp/parameter_efficient_pretraining || exit

export WANDB_PROJECT="SLTrain_Restarts"

 
torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --model_name 350m_low-rank_1.5e-3_fp32 \
    --model_config configs/llama_350m.json \
    --lr 0.0015 \
    --rank 256 \
    --peft_model low-rank \
    --optimizer adamW \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --dtype fp32 \
    --eval_every 1000 \
    --save_every 99999 \
    --lora_alpha 16 \
    --cycle_length 60000 \
    --weight_decay 0.0 \
    --scheduler cosine \
    --seed 42  \
    --dataset_path /code/hongpaul-sandbox/parameter_efficient_pretraining/c4/en
 

# /usr/bin/shutdown
