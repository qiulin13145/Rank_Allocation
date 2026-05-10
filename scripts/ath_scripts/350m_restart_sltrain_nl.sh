#!/bin/bash

#SBATCH --time=72:00:00
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


# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --model_name 350_sltrain_restart_nl_1e-2_a_32 \
    --model_config configs/llama_350m.json \
    --lr 1e-2 \
    --peft_model sltrain \
    --optimizer adamw_beta \
    --rank 256 \
    --sp_ratio 0.1 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 32 \
    --save_every 99999 \
    --cycle_length 200 \
    --weight_decay 0.0 \
    --scheduler cosine_quick_recovery\
    --restart_warmup_steps 10 \
    --seed 42  \
    --max_length 256 \