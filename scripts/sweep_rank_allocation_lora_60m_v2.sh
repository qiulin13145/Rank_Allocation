#!/bin/bash

set -uo pipefail

# Sweep for RankAllocationLoRA 60M — focused on delta/top_k and timing/tail params.
# Runs are sequential. Each run gets its own log file, and hyperparameters/results
# are written to CSV files under log/rank_allocation_lora_sweep_<timestamp>/.

export WANDB_PROJECT="${WANDB_PROJECT:-rank_allocation_lora_sweep}"

if [ -n "${WANDB_KEY:-}" ]; then
    python -m wandb login "$WANDB_KEY"
fi

SWEEP_MAX_RUNS="${SWEEP_MAX_RUNS:-30}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-log/rank_allocation_lora_sweep_${STAMP}}"
mkdir -p "$OUT_DIR"

MANIFEST="$OUT_DIR/manifest.csv"
SUMMARY="$OUT_DIR/summary.csv"

echo "run_id,model_name,delta,top_k,start_step,interval,tail_threshold,log_path" > "$MANIFEST"
echo "run_id,model_name,delta,top_k,start_step,interval,tail_threshold,exit_code,last_eval_loss,last_eval_ppl,log_path" > "$SUMMARY"

BASE_ARGS=(
    --model_config configs/llama_60m.json
    --lr 0.003
    --peft_model rank_allocation_lora
    --optimizer adamW
    --rank 128
    --batch_size 64
    --total_batch_size 512
    --num_training_steps 11000
    --warmup_steps 1100
    --dtype bfloat16
    --eval_every 1000
    --lora_alpha 32
    --save_every 50000
    --weight_decay 0.0
    --cycle_length 500
    --scheduler cosine_quick_recovery
    --restart_warmup_steps 10
    --rank_allocation_min_ratio 0.1875
    --rank_allocation_max_ratio 0.3125
    --rank_allocation_hysteresis 0.0
    --rank_allocation_probe_rank 8
    --rank_allocation_probe_sigma 1e-3
    --rank_allocation_credit_sample_interval 5
    --rank_allocation_credit_beta 0.95
    --rank_allocation_probe_beta 0.9
    --rank_allocation_extra_init_std 1e-4
    --rank_allocation_report_dir log/rank_allocation_reports
    --dataset_path /data/datasets/c4/en
    --seed 42
)

# Format: delta top_k start_step interval tail_threshold tag
#
# Group 1 (high priority): sweep delta / top_k.
# Group 2 (lower priority): sweep start_step / interval / tail_threshold.
COMBOS=(
    # ---- group 1: delta × top_k ----
    "4  2  2000 500 0.1  d4_k2"
    "8  2  2000 500 0.1  d8_k2_base"
    "16 2  2000 500 0.1  d16_k2"
    "32 2  2000 500 0.1  d32_k2"
    "8  4  2000 500 0.1  d8_k4"
    "16 4  2000 500 0.1  d16_k4"
    "4  4  2000 500 0.1  d4_k4"
    "32 4  2000 500 0.1  d32_k4"

    # ---- group 2: start_step / interval / tail_threshold ----
    "8  2  1000 500  0.1   s1000"
    "8  2  4000 500  0.1   s4000"
    "8  2  2000 1000 0.1   i1000"
    "8  2  2000 2000 0.1   i2000"
    "8  2  2000 500  0.05  t005"
    "8  2  2000 500  0.2   t02"
    "8  2  2000 500  0.5   t05"
)

extract_last_eval() {
    local log_path="$1"
    local line
    line="$(grep -E "Eval loss and perplexity at step" "$log_path" | tail -n 1 || true)"
    if [ -z "$line" ]; then
        echo ","
        return
    fi
    echo "$line" | sed -E 's/.*: ([0-9.eE+-]+), ([0-9.eE+-]+).*/\1,\2/'
}

run_index=0
for combo in "${COMBOS[@]}"; do
    if [ "$run_index" -ge "$SWEEP_MAX_RUNS" ]; then
        break
    fi

    read -r delta top_k start_step interval tail_threshold tag <<< "$combo"
    run_id="$(printf "%02d" "$((run_index + 1))")"
    model_name="ralora60m_${run_id}_${tag}_d${delta}_k${top_k}_s${start_step}_i${interval}_t${tail_threshold}"
    log_path="$OUT_DIR/${run_id}_${tag}.log"

    echo "[$(date)] Starting run ${run_id}/${SWEEP_MAX_RUNS}: ${model_name}"
    echo "${run_id},${model_name},${delta},${top_k},${start_step},${interval},${tail_threshold},${log_path}" >> "$MANIFEST"

    torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
        --model_name "$model_name" \
        "${BASE_ARGS[@]}" \
        --rank_allocation_delta "$delta" \
        --rank_allocation_top_k "$top_k" \
        --rank_allocation_start_step "$start_step" \
        --rank_allocation_interval "$interval" \
        --rank_allocation_tail_threshold "$tail_threshold" \
        > "$log_path" 2>&1

    exit_code="$?"
    eval_pair="$(extract_last_eval "$log_path")"
    echo "${run_id},${model_name},${delta},${top_k},${start_step},${interval},${tail_threshold},${exit_code},${eval_pair},${log_path}" >> "$SUMMARY"
    echo "[$(date)] Finished run ${run_id}: exit_code=${exit_code}, eval=${eval_pair}, log=${log_path}"

    run_index="$((run_index + 1))"
done

echo "Sweep complete."
echo "Manifest: $MANIFEST"
echo "Summary : $SUMMARY"
