#!/bin/bash

set -uo pipefail

# Manual overnight sweep for RankAllocationLoRA 60M.
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

echo "run_id,model_name,delta,top_k,min_ratio,max_ratio,hysteresis,probe_rank,probe_sigma,credit_sample_interval,credit_beta,probe_beta,start_step,log_path" > "$MANIFEST"
echo "run_id,model_name,delta,top_k,min_ratio,max_ratio,hysteresis,probe_rank,exit_code,last_eval_loss,last_eval_ppl,log_path" > "$SUMMARY"

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
    --rank_allocation_probe_sigma 1e-3
    --rank_allocation_credit_sample_interval 20
    --rank_allocation_credit_beta 0.95
    --rank_allocation_probe_beta 0.9
    --rank_allocation_start_step 1100
    --rank_allocation_report_dir log/rank_allocation_reports
    --dataset_path /data/datasets/c4/en
    --seed 42
)

# Format:
# delta top_k min_ratio max_ratio hysteresis probe_rank tag
#
# Priority: first sweep delta/top_k aggressively, then test safety bounds,
# hysteresis, and probe_rank around the best default-ish region.
COMBOS=(
    "8 4 0.125 0.5 0.10 1 default"
    "16 4 0.125 0.5 0.10 1 delta16_top4"
    "8 8 0.125 0.5 0.10 1 delta8_top8"
    "16 8 0.125 0.5 0.10 1 delta16_top8"
    "32 4 0.125 0.5 0.10 1 delta32_top4"
    "32 8 0.125 0.5 0.10 1 delta32_top8"
    "8 2 0.125 0.5 0.10 1 delta8_top2"
    "16 2 0.125 0.5 0.10 1 delta16_top2"
    "32 2 0.125 0.5 0.10 1 delta32_top2"
    "8 12 0.125 0.5 0.10 1 delta8_top12"
    "16 12 0.125 0.5 0.10 1 delta16_top12"
    "24 4 0.125 0.5 0.10 1 delta24_top4"
    "24 8 0.125 0.5 0.10 1 delta24_top8"
    "24 12 0.125 0.5 0.10 1 delta24_top12"
    "16 4 0.0625 0.5 0.10 1 min00625"
    "16 8 0.0625 0.5 0.10 1 min00625_top8"
    "16 4 0.125 0.75 0.10 1 max075"
    "16 8 0.125 0.75 0.10 1 max075_top8"
    "16 4 0.25 0.5 0.10 1 min025"
    "16 8 0.25 0.5 0.10 1 min025_top8"
    "16 4 0.125 0.5 0.00 1 hyst000"
    "16 8 0.125 0.5 0.00 1 hyst000_top8"
    "16 4 0.125 0.5 0.25 1 hyst025"
    "16 8 0.125 0.5 0.25 1 hyst025_top8"
    "16 4 0.125 0.5 0.10 2 probe2"
    "16 8 0.125 0.5 0.10 2 probe2_top8"
    "16 4 0.125 0.5 0.10 4 probe4"
    "16 8 0.125 0.5 0.10 4 probe4_top8"
    "8 16 0.125 0.5 0.10 1 delta8_top16"
    "16 16 0.125 0.5 0.10 1 delta16_top16"
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

    read -r delta top_k min_ratio max_ratio hysteresis probe_rank tag <<< "$combo"
    run_id="$(printf "%02d" "$((run_index + 1))")"
    model_name="ralora60m_${run_id}_${tag}_d${delta}_k${top_k}_min${min_ratio}_max${max_ratio}_h${hysteresis}_p${probe_rank}"
    log_path="$OUT_DIR/${run_id}_${tag}.log"

    echo "[$(date)] Starting run ${run_id}/${SWEEP_MAX_RUNS}: ${model_name}"
    echo "${run_id},${model_name},${delta},${top_k},${min_ratio},${max_ratio},${hysteresis},${probe_rank},1e-3,20,0.95,0.9,1100,${log_path}" >> "$MANIFEST"

    torchrun --standalone --nproc_per_node 8 torchrun_main_DDP.py \
        --model_name "$model_name" \
        "${BASE_ARGS[@]}" \
        --rank_allocation_delta "$delta" \
        --rank_allocation_top_k "$top_k" \
        --rank_allocation_min_ratio "$min_ratio" \
        --rank_allocation_max_ratio "$max_ratio" \
        --rank_allocation_hysteresis "$hysteresis" \
        --rank_allocation_probe_rank "$probe_rank" \
        > "$log_path" 2>&1

    exit_code="$?"
    eval_pair="$(extract_last_eval "$log_path")"
    echo "${run_id},${model_name},${delta},${top_k},${min_ratio},${max_ratio},${hysteresis},${probe_rank},${exit_code},${eval_pair},${log_path}" >> "$SUMMARY"
    echo "[$(date)] Finished run ${run_id}: exit_code=${exit_code}, eval=${eval_pair}, log=${log_path}"

    run_index="$((run_index + 1))"
done

echo "Sweep complete."
echo "Manifest: $MANIFEST"
echo "Summary : $SUMMARY"
