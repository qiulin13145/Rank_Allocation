#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=eff
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
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
cd /users/3/glent007/workspace/pretrain2/parameter_efficient_pretraining || exit

export WANDB_PROJECT="SLTrain_Restarts"

torchrun --standalone --nproc_per_node 1 torchrun_main_DDP.py \
    --model_name 60m_swan_v2-both-all_lr=2e-2 \
    --model_config configs/llama_60m.json \
    --optimizer swan \
    --swan_version v2_both_all \
    --lr 2e-2 \
    --momentum 0.9 \
    --adam_beta_1 0.9 \
    --adam_beta_2 0.999 \
    --weight_decay 0.0 \
    --peft_model full-rank \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 11000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --cycle_length 11000 \
    --save_every 99999 \
    --seed 42  \
    --scheduler cosine \
    --warmup_steps 1100 \


#   --galore_use_nl 
#    --model_name 60m_adam-svd_5e-3_0.9-0.99_biascorr_new \
#   --muon_on svd_adamw_pre_ns \

  #  --swan_only_mpl_att \

  # 60m_swan_original_mpl_att_a=0.05_lr=2e-2