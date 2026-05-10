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
cd /users/3/glent007/workspace/pretrain/parameter_efficient_pretraining || exit
export WANDB_PROJECT="SLTrain_Restarts"


 
torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --model_config configs/llama_60m.json \
    --model_name full-rank_350m__beta_5e-3_NL_GS_appolo-code \
    --lr 5e-3 \
    --peft_model full-rank \
    --optimizer adam_beta \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 60000 \
    --seed 42  \
    --scheduler cosine \
    --cycle_length 60000 \
    --grad_clipping 0 \
    --scale_type channel \

# -- disable_nl
# -- disable_grad_scaling
    

 # !!! CHANGE TO 512

    #lr 0.005
    
    #--grad_clipping 0.1 \

# adamw_beta
