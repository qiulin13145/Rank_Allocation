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

 

# LLaMA-1B
torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --keep_only_last_model \
    --model_name restart_1b_sltrain_cl200 \
    --model_config configs/llama_1b.json \
    --lr 0.005 \
    --peft_model full-rank \ ???
    --optimizer adamW \
    --rank 512 \
    --sp_ratio 0.1 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 140000 \
    --warmup_steps 14000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --lora_alpha 16 \
    --save_every 5000 \
    --cycle_length 200 \
    --weight_decay 0.0 \
    --scheduler cosine_quick_recovery\
    --restart_warmup_steps 10 \
    --seed 42  \
    --workers 8 \
     
     

#export OMP_NUM_THREADS=32
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
#Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration, 
#which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has
# flow control causing later iterations to have unused parameters. (function operator())
# /usr/bin/shutdown