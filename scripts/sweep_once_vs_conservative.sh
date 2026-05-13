#!/bin/bash

set -uo pipefail

# Two-layer sweep comparing once vs. conservative rank allocation.
# Layer 1 ("once"):  --rank_allocation_once, interval=500
# Layer 2 ("cons"):  no --once flag,           interval=1000
#
# Each layer sweeps delta × top_k (top_k priority: 16→32→64→128, then delta: 8→16→32).

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-log/sweep_once_cons_${STAMP}}"
mkdir -p "$OUT_DIR"

MANIFEST="$OUT_DIR/manifest.csv"
SUMMARY="$OUT_DIR/summary.csv"

echo "run_id,layer,delta,top_k,model_name,log_path" > "$MANIFEST"
echo "run_id,layer,delta,top_k,model_name,exit_code,last_eval_loss,last_eval_ppl,log_path" > "$SUMMARY"

SHARED_ARGS=(
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
    --rank_allocation_min_ratio 0.1
    --rank_allocation_max_ratio 0.5
    --rank_allocation_hysteresis 0.0
    --rank_allocation_probe_rank 8
    --rank_allocation_probe_sigma 1e-3
    --rank_allocation_credit_sample_interval 5
    --rank_allocation_credit_beta 0.95
    --rank_allocation_probe_beta 0.9
    --rank_allocation_start_step 4000
    --rank_allocation_tail_threshold 0.3
    --rank_allocation_extra_init_std 1e-4
    --rank_allocation_report_dir log/rank_allocation_reports
    --dataset_path /data/datasets/c4/en
    --seed 42
)

# Format: delta top_k layer_tag
# Priority: top_k first (16→32→64→128), then delta (8→16→32) within each top_k.
COMBOS=(
    # ===== Layer 1: once (interval=500) =====
    "8  16  once"
    "16 16  once"
    "32 16  once"
    "8  32  once"
    "16 32  once"
    "32 32  once"
    "8  64  once"
    "16 64  once"
    "32 64  once"
    "8  128 once"
    "16 128 once"
    "32 128 once"

    # ===== Layer 2: conservative (interval=1000, no --once) =====
    "8  16  cons"
    "16 16  cons"
    "32 16  cons"
    "8  32  cons"
    "16 32  cons"
    "32 32  cons"
    "8  64  cons"
    "16 64  cons"
    "32 64  cons"
    "8  128 cons"
    "16 128 cons"
    "32 128 cons"
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
    read -r delta top_k layer <<< "$combo"
    run_id="$(printf "%02d" "$((run_index + 1))")"

    if [ "$layer" = "once" ]; then
        export WANDB_PROJECT="rank_allocation_sweep_once"
        model_name="once_k${top_k}_d${delta}_${STAMP}"
        interval=500
        once_flag=(--rank_allocation_once)
    else
        export WANDB_PROJECT="rank_allocation_sweep_cons"
        model_name="cons_k${top_k}_d${delta}_${STAMP}"
        interval=1000
        once_flag=()
    fi

    log_path="$OUT_DIR/${run_id}_${layer}_k${top_k}_d${delta}.log"

    echo "[$(date)] [${layer}] Starting run ${run_id}: ${model_name}  (delta=${delta}, top_k=${top_k}, interval=${interval})"
    echo "${run_id},${layer},${delta},${top_k},${model_name},${log_path}" >> "$MANIFEST"

    python -m wandb login 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 2>/dev/null

    torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
        --model_name "$model_name" \
        "${SHARED_ARGS[@]}" \
        --rank_allocation_delta "$delta" \
        --rank_allocation_top_k "$top_k" \
        --rank_allocation_interval "$interval" \
        "${once_flag[@]}" \
        > "$log_path" 2>&1

    exit_code="$?"
    eval_pair="$(extract_last_eval "$log_path")"
    echo "${run_id},${layer},${delta},${top_k},${model_name},${exit_code},${eval_pair},${log_path}" >> "$SUMMARY"
    echo "[$(date)] [${layer}] Finished run ${run_id}: exit=${exit_code}, eval=${eval_pair}"

    run_index="$((run_index + 1))"
done

echo ""
echo "Sweep complete. ${run_index} runs total."
echo "Manifest : $MANIFEST"
echo "Summary  : $SUMMARY"
