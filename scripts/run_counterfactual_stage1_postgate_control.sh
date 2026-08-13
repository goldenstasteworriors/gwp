#!/usr/bin/env bash
set -euo pipefail

# This watcher is intentionally separate from the stage-one pipeline. It waits
# for the held-out mechanism gate, then and only then evaluates policy control.

PROJECT=${GWP_CF_PROJECT:-/home/ubuntu/project/WM/gwp-action-video-consequence-alignment}
BRANCH=${GWP_CF_BRANCH:-codex/action-video-consequence-alignment-20260813}
TARGET_COMMIT=${GWP_CF_TARGET_COMMIT:?GWP_CF_TARGET_COMMIT must pin audited evaluation code}
RESULT_ROOT=${GWP_CF_RESULT_ROOT:-/home/ubuntu/project/WM/gwp-action-video-results/counterfactual_mask_stage1_v3_pipeline_30k_ema}
BASELINE_ROOT=${GWP_CF_BASELINE_ROOT:-/home/ubuntu/project/WM/gwp-action-video-results/libero_standard4_gwp05_official_30k_4gpu_2cam}
BASELINE_CHECKPOINT=${GWP_CF_CHECKPOINT:-$BASELINE_ROOT/models/checkpoint_epoch_2_step_30000/transformer_ema}
DATASET=${GWP_CF_DATASET:-/home/ubuntu/project/WM/datasets/LIBERO_LeRobot_v3/libero_spatial}
NORM_STATS=${GWP_CF_NORM_STATS:-/home/ubuntu/project/WM/datasets/LIBERO_LeRobot_v3/norm_stats_wam_48_standard4.json}
WAN=${GWP_CF_WAN:-/home/ubuntu/project/WM/giga-world-policy/weights/Wan2.2-TI2V-5B-Diffusers}
VAE=$WAN/vae
SPLIT_ROOT=${GWP_CF_SPLIT_ROOT:-/home/ubuntu/project/WM/gwp-action-video-results/counterfactual_mask_splits/seed_20260813/libero_spatial_20train_5heldout}
POLICY_PYTHON=${GWP_CF_POLICY_PYTHON:-/home/ubuntu/miniconda3/envs/gigaworld-policy/bin/python}
EVAL_PYTHON=${GWP_CF_EVAL_PYTHON:-/home/ubuntu/miniconda3/envs/gigaworld-libero-eval/bin/python}
CONTROL_ROOT=${GWP_CF_CONTROL_ROOT:-$RESULT_ROOT/control_eval_spatial_10x10}
PAIR_AUDIT=$RESULT_ROOT/pair_audit.json
PIPELINE_PID_FILE=$RESULT_ROOT/pipeline.pid
GPU_IDS=(2 3 4)
PORTS=(11720 11721 11722)

status() {
    printf '%s\n' "$1" > "$CONTROL_ROOT/status"
}

cleanup() {
    local code=$?
    if [ "$code" -ne 0 ]; then
        cat "$CONTROL_ROOT/status" > "$CONTROL_ROOT/failed_stage" 2>/dev/null || true
        status failed
    fi
    printf '%s\n' "$code" > "$CONTROL_ROOT/exit_code"
}

wait_for_gpus() {
    local busy gpu_id
    while true; do
        busy=0
        for gpu_id in "${GPU_IDS[@]}"; do
            if nvidia-smi -i "$gpu_id" --query-compute-apps=pid \
                --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
                busy=1
                break
            fi
        done
        if [ "$busy" -eq 0 ]; then
            return
        fi
        sleep 60
    done
}

mkdir -p "$CONTROL_ROOT/logs"
trap cleanup EXIT
test -f "$PIPELINE_PID_FILE"
test ! -e "$CONTROL_ROOT/baseline"
test ! -e "$CONTROL_ROOT/matched"
test ! -e "$CONTROL_ROOT/full"
test ! -e "$CONTROL_ROOT/control_audit.json"

status waiting_stage1_pipeline
PIPELINE_PID=$(cat "$PIPELINE_PID_FILE")
[[ "$PIPELINE_PID" =~ ^[1-9][0-9]*$ ]]
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
    sleep 60
done
test "$(cat "$RESULT_ROOT/status")" = "complete"
test "$(cat "$RESULT_ROOT/exit_code")" = "0"
test -f "$PAIR_AUDIT"

status validating_mechanism_gate
"$POLICY_PYTHON" -B -c 'import json,sys; value=json.load(open(sys.argv[1])); assert value["audit_status"]=="passed"; assert value["mechanism_gate"]["status"] in {"passed","failed"}; assert value["control_evaluation_allowed"] is (value["mechanism_gate"]["status"]=="passed")' "$PAIR_AUDIT"
MECHANISM_GATE=$("$POLICY_PYTHON" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["mechanism_gate"]["status"])' "$PAIR_AUDIT")
if [ "$MECHANISM_GATE" != "passed" ]; then
    status stopped_mechanism_gate_failed
    exit 0
fi

status syncing_evaluation_code
test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=no)"
git -C "$PROJECT" fetch alignment-backup "$BRANCH"
git -C "$PROJECT" merge --ff-only "$TARGET_COMMIT"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$TARGET_COMMIT"
test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=no)"
git -C "$PROJECT" rev-parse HEAD > "$CONTROL_ROOT/code_commit.txt"
cd "$PROJECT"

mapfile -t MATCHED_CHECKPOINTS < <(
    find "$RESULT_ROOT/matched_200step/models" -type d \
        -path '*/checkpoint_*_step_200/transformer' -print
)
mapfile -t FULL_CHECKPOINTS < <(
    find "$RESULT_ROOT/full_200step/models" -type d \
        -path '*/checkpoint_*_step_200/transformer' -print
)
test "${#MATCHED_CHECKPOINTS[@]}" -eq 1
test "${#FULL_CHECKPOINTS[@]}" -eq 1
MATCHED_CHECKPOINT=${MATCHED_CHECKPOINTS[0]}
FULL_CHECKPOINT=${FULL_CHECKPOINTS[0]}

AVAILABLE_BYTES=$(df -B1 --output=avail "$CONTROL_ROOT" | tail -n 1 | tr -d ' ')
MINIMUM_BYTES=$((20 * 1024 * 1024 * 1024))
[[ "$AVAILABLE_BYTES" =~ ^[0-9]+$ ]]
test "$AVAILABLE_BYTES" -ge "$MINIMUM_BYTES"

status waiting_gpus
wait_for_gpus
status evaluating_baseline_matched_full

run_bundle() {
    local name=$1 checkpoint=$2 gpu_id=$3 port=$4
    "$POLICY_PYTHON" -B scripts/evaluate_libero_policy_bundle.py \
        --checkpoint "$checkpoint" \
        --vae "$VAE" \
        --norm-stats "$NORM_STATS" \
        --dataset-root "$DATASET" \
        --suite libero_spatial \
        --episodes-file "$SPLIT_ROOT/heldout_episodes.txt" \
        --output-dir "$CONTROL_ROOT/$name" \
        --policy-python "$POLICY_PYTHON" \
        --open-loop-python "$EVAL_PYTHON" \
        --closed-loop-python "$EVAL_PYTHON" \
        --gpu-id "$gpu_id" \
        --port "$port" \
        --rollouts-per-task 10 \
        --max-steps 280 \
        --replan-steps 10 \
        --num-inference-steps 10 \
        --model-dtype bf16 \
        --seed 20260813 \
        > "$CONTROL_ROOT/logs/${name}_bundle.log" 2>&1
}

run_bundle baseline "$BASELINE_CHECKPOINT" "${GPU_IDS[0]}" "${PORTS[0]}" &
BASELINE_PID=$!
run_bundle matched "$MATCHED_CHECKPOINT" "${GPU_IDS[1]}" "${PORTS[1]}" &
MATCHED_PID=$!
run_bundle full "$FULL_CHECKPOINT" "${GPU_IDS[2]}" "${PORTS[2]}" &
FULL_PID=$!

set +e
wait "$BASELINE_PID"; BASELINE_CODE=$?
wait "$MATCHED_PID"; MATCHED_CODE=$?
wait "$FULL_PID"; FULL_CODE=$?
set -e
if [ "$BASELINE_CODE" -ne 0 ] || [ "$MATCHED_CODE" -ne 0 ] || [ "$FULL_CODE" -ne 0 ]; then
    printf 'baseline=%s matched=%s full=%s\n' \
        "$BASELINE_CODE" "$MATCHED_CODE" "$FULL_CODE" \
        > "$CONTROL_ROOT/bundle_exit_codes.txt"
    exit 1
fi

status comparing_open_loop
"$POLICY_PYTHON" -B scripts/compare_libero_open_loop.py \
    --left "$CONTROL_ROOT/full/open_loop_heldout/summary.json" \
    --right "$CONTROL_ROOT/matched/open_loop_heldout/summary.json" \
    --left-label full --right-label matched \
    --output "$CONTROL_ROOT/full_vs_matched_open.json" \
    > "$CONTROL_ROOT/logs/full_vs_matched_open.log" 2>&1
"$POLICY_PYTHON" -B scripts/compare_libero_open_loop.py \
    --left "$CONTROL_ROOT/full/open_loop_heldout/summary.json" \
    --right "$CONTROL_ROOT/baseline/open_loop_heldout/summary.json" \
    --left-label full --right-label baseline \
    --output "$CONTROL_ROOT/full_vs_baseline_open.json" \
    > "$CONTROL_ROOT/logs/full_vs_baseline_open.log" 2>&1

status comparing_closed_loop
"$POLICY_PYTHON" -B scripts/compare_libero_paired_rollouts.py \
    --left "$CONTROL_ROOT/full/closed_loop/summary.json" \
    --right "$CONTROL_ROOT/matched/closed_loop/summary.json" \
    --left-label full --right-label matched \
    --output "$CONTROL_ROOT/full_vs_matched_closed.json" \
    > "$CONTROL_ROOT/logs/full_vs_matched_closed.log" 2>&1
"$POLICY_PYTHON" -B scripts/compare_libero_paired_rollouts.py \
    --left "$CONTROL_ROOT/full/closed_loop/summary.json" \
    --right "$CONTROL_ROOT/baseline/closed_loop/summary.json" \
    --left-label full --right-label baseline \
    --output "$CONTROL_ROOT/full_vs_baseline_closed.json" \
    > "$CONTROL_ROOT/logs/full_vs_baseline_closed.log" 2>&1

status auditing_control_evaluations
"$POLICY_PYTHON" -B scripts/audit_counterfactual_control_evaluations.py \
    --pair-audit "$PAIR_AUDIT" \
    --baseline-checkpoint "$BASELINE_CHECKPOINT" \
    --baseline-bundle "$CONTROL_ROOT/baseline/manifest.json" \
    --matched-bundle "$CONTROL_ROOT/matched/manifest.json" \
    --full-bundle "$CONTROL_ROOT/full/manifest.json" \
    --full-vs-matched-open "$CONTROL_ROOT/full_vs_matched_open.json" \
    --full-vs-baseline-open "$CONTROL_ROOT/full_vs_baseline_open.json" \
    --full-vs-matched-closed "$CONTROL_ROOT/full_vs_matched_closed.json" \
    --full-vs-baseline-closed "$CONTROL_ROOT/full_vs_baseline_closed.json" \
    --output "$CONTROL_ROOT/control_audit.json" \
    > "$CONTROL_ROOT/logs/control_audit.log" 2>&1

status complete
