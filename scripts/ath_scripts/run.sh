#!/bin/bash

#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:a100:4
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
cd /users/3/glent007/workspace/pretrain/parameter_efficient_pretraining || exit
export WANDB_PROJECT="Trash"

export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nproc_per_node 1 torchrun_debug.py \
    --keep_only_last_model \
    --model_config configs/llama_60m.json \
    --model_name test_xx \
    --lr 0.01 \
    --peft_model full-rank \
    --optimizer adamw \
    --batch_size 512 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --warmup_steps 1100 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 100 \
    --seed 42  \
    --scheduler cosine \
    --cycle_length 11000 \
    --workers 8 \
 
    #     --hf_dataset \
    #--start_tokenizing_idx 200000

    # start_tokenizing_idx * workers < continue_from_global_step (48*8 = 384 < 400 )
 
#    --continue_from /users/3/glent007/workspace/pretrain/parameter_efficient_pretraining/checkpoints/llama_60m-2025-02-20-04-23-19/model_100 \
 
#--continue_from /users/3/glent007/workspace/pretrain/parameter_efficient_pretraining/checkpoints/llama_60m-2025-02-19-15-40-34/model_100 \

#     --max_lenght 2 \

# full rank and galore save safetensors ????????
