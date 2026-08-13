#!/usr/bin/env bash
set -euo pipefail

# One-shot, audited stage-one pipeline. It waits for the official 30k baseline
# and verified raw demonstrations, then runs v3 smoke, cache generation,
# matched/full training, held-out evaluation, and the pair audit.

PROJECT=${GWP_CF_PROJECT:-/home/ubuntu/project/WM/gwp-action-video-consequence-alignment}
RESULT_ROOT=${GWP_CF_RESULT_ROOT:-/home/ubuntu/project/WM/gwp-action-video-results/counterfactual_mask_stage1_v3_pipeline_30k_ema}
BASELINE_ROOT=${GWP_CF_BASELINE_ROOT:-/home/ubuntu/project/WM/gwp-action-video-results/libero_standard4_gwp05_official_30k_4gpu_2cam}
CHECKPOINT=${GWP_CF_CHECKPOINT:-$BASELINE_ROOT/models/checkpoint_epoch_2_step_30000/transformer_ema}
BASELINE_EVAL_PID_FILE=${GWP_CF_BASELINE_EVAL_PID_FILE:-$BASELINE_ROOT/auto_eval.pid}
BASELINE_EVAL_OUTPUT=${GWP_CF_BASELINE_EVAL_OUTPUT:-$BASELINE_ROOT/closed_loop_standard4_10x10_ema}
RAW_DEMOS=${GWP_CF_RAW_DEMOS:-/home/ubuntu/project/WM/datasets/LIBERO_raw/libero_spatial}
DATASET=${GWP_CF_DATASET:-/home/ubuntu/project/WM/datasets/LIBERO_LeRobot_v3/libero_spatial}
NORM_STATS=${GWP_CF_NORM_STATS:-/home/ubuntu/project/WM/datasets/LIBERO_LeRobot_v3/norm_stats_wam_48_standard4.json}
WAN=${GWP_CF_WAN:-/home/ubuntu/project/WM/giga-world-policy/weights/Wan2.2-TI2V-5B-Diffusers}
VAE=$WAN/vae
SPLIT_ROOT=${GWP_CF_SPLIT_ROOT:-/home/ubuntu/project/WM/gwp-action-video-results/counterfactual_mask_splits/seed_20260813/libero_spatial_20train_5heldout}
SMOKE_CACHE=${GWP_CF_SMOKE_CACHE:-/home/ubuntu/project/WM/gwp-action-video-results/counterfactual_mask_cache_smoke_op55_v3}
POLICY_PYTHON=${GWP_CF_POLICY_PYTHON:-/home/ubuntu/miniconda3/envs/gigaworld-policy/bin/python}
EVAL_PYTHON=${GWP_CF_EVAL_PYTHON:-/home/ubuntu/miniconda3/envs/gigaworld-libero-eval/bin/python}
GPU_IDS=(2 3 4 5)
RENDER_GPU=2
PORT=${GWP_CF_POLICY_PORT:-11640}

ACTIONS_TRAIN=$RESULT_ROOT/actions_train
ACTIONS_HELDOUT=$RESULT_ROOT/actions_heldout
CACHE_TRAIN=$RESULT_ROOT/cache_train
CACHE_HELDOUT=$RESULT_ROOT/cache_heldout
MATCHED_OUTPUT=$RESULT_ROOT/matched_200step
FULL_OUTPUT=$RESULT_ROOT/full_200step
PREFLIGHT_REPORT=$RESULT_ROOT/preflight_report.json
MATCHED_EVAL=$RESULT_ROOT/matched_heldout.json
FULL_EVAL=$RESULT_ROOT/full_heldout.json
PAIR_AUDIT=$RESULT_ROOT/pair_audit.json
VIDEO_SPLIT_AUDIT=$RESULT_ROOT/video_split_audit.json
SERVER_PID=""

status() {
    printf '%s\n' "$1" > "$RESULT_ROOT/status"
}

cleanup() {
    local code=$?
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ "$code" -ne 0 ]; then
        cat "$RESULT_ROOT/status" > "$RESULT_ROOT/failed_stage" 2>/dev/null || true
        status failed
    fi
    printf '%s\n' "$code" > "$RESULT_ROOT/exit_code"
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

process_is_baseline_evaluator() {
    local pid=$1
    local command
    if [ ! -r "/proc/$pid/cmdline" ]; then
        return 1
    fi
    command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    [[ "$command" == *"evaluate_libero_standard4_parallel.py"* ]] && \
        [[ "$command" == *"$BASELINE_EVAL_OUTPUT"* ]]
}

mkdir -p "$RESULT_ROOT/logs"
trap cleanup EXIT
cd "$PROJECT"

status validating_static_inputs
test -e "$PROJECT/.git"
test -z "$(git status --porcelain --untracked-files=no)"
test -d "$RAW_DEMOS"
test -d "$DATASET"
test -f "$NORM_STATS"
test -d "$WAN"
test -d "$VAE"
test -f "$BASELINE_EVAL_PID_FILE"
test -f "$SPLIT_ROOT/train_episodes.txt"
test -f "$SPLIT_ROOT/heldout_episodes.txt"
test -f "$SMOKE_CACHE/manifest.json"
git rev-parse HEAD > "$RESULT_ROOT/code_commit.txt"

for path in "$ACTIONS_TRAIN" "$ACTIONS_HELDOUT" "$CACHE_TRAIN" \
    "$CACHE_HELDOUT" "$MATCHED_OUTPUT" "$FULL_OUTPUT" "$PREFLIGHT_REPORT" \
    "$MATCHED_EVAL" "$FULL_EVAL" "$PAIR_AUDIT"; do
    test ! -e "$path"
done

status waiting_raw_demo_downloads
until "$POLICY_PYTHON" -B scripts/verify_libero_raw_demos.py \
    --root "$RAW_DEMOS" --size-only \
    > "$RESULT_ROOT/logs/raw_demo_size_check.json" 2>&1; do
    sleep 60
done

status hashing_raw_demos
"$POLICY_PYTHON" -B scripts/verify_libero_raw_demos.py \
    --root "$RAW_DEMOS" \
    > "$RESULT_ROOT/raw_demo_verification.json"

status auditing_selected_videos
if [ -f "$VIDEO_SPLIT_AUDIT" ]; then
    "$POLICY_PYTHON" -B -c 'import hashlib,json,sys; from pathlib import Path; report=json.loads(Path(sys.argv[1]).read_text()); dataset=Path(sys.argv[2]).resolve(); files=[Path(value).resolve() for value in sys.argv[3:]]; digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest(); sources=report["episode_sources"]; assert report["status"]=="passed"; assert Path(report["dataset_root"]).resolve()==dataset; assert report["num_episodes"]==250 and report["horizons"]==[0,12,24,36,48]; assert report["view_keys"]==["observation.images.image","observation.images.wrist_image"]; assert report["video_backend"]=="pyav" and report["zero_or_constant_fallbacks"]==0; assert [Path(value["path"]).resolve() for value in sources]==files; assert [value["sha256"] for value in sources]==[digest(path) for path in files]' \
        "$VIDEO_SPLIT_AUDIT" "$DATASET" \
        "$SPLIT_ROOT/train_episodes.txt" "$SPLIT_ROOT/heldout_episodes.txt"
else
    "$POLICY_PYTHON" -B scripts/audit_lerobot_video_split.py \
        --dataset-root "$DATASET" \
        --episodes-file "$SPLIT_ROOT/train_episodes.txt" \
        --episodes-file "$SPLIT_ROOT/heldout_episodes.txt" \
        --report "$VIDEO_SPLIT_AUDIT" \
        > "$RESULT_ROOT/logs/video_split_audit.log" 2>&1
fi

status waiting_checkpoint
while [ ! -d "$CHECKPOINT" ]; do
    sleep 60
done

status waiting_baseline_evaluation
BASELINE_EVAL_PID=$(cat "$BASELINE_EVAL_PID_FILE")
[[ "$BASELINE_EVAL_PID" =~ ^[1-9][0-9]*$ ]]
while process_is_baseline_evaluator "$BASELINE_EVAL_PID"; do
    sleep 60
done
"$POLICY_PYTHON" -B -c 'import json,sys; from pathlib import Path; summary=json.loads(Path(sys.argv[1]).read_text()); expected=Path(sys.argv[2]).resolve(); assert Path(summary["checkpoint"]).resolve()==expected; assert int(summary["num_rollouts"])==400; assert set(summary["suites"])=={"libero_spatial","libero_object","libero_goal","libero_10"}; assert all(int(value["num_rollouts"])==100 for value in summary["suites"].values())' \
    "$BASELINE_EVAL_OUTPUT/summary.json" "$CHECKPOINT"

status waiting_gpus_for_smoke
wait_for_gpus
status running_v3_smoke
(
    export GWP_LIBERO_DATA_PATH="$DATASET"
    export GWP_NORM_STATS_PATH="$NORM_STATS"
    export GWP_WAN_PRETRAINED="$WAN"
    export GWP05_TRANSFORMER_PRETRAINED="$CHECKPOINT"
    export GWP05_PROJECT_DIR="$RESULT_ROOT/smoke_train"
    export GWP_T5_EMBEDDING_DIR="$DATASET/t5_embedding"
    export GWP_VIEW_KEYS=observation.images.image,observation.images.wrist_image
    export GWP_VIEW_CONCAT_MODE=horizontal
    export GWP_GPU_IDS=2,3,4,5
    export GWP_BATCH_SIZE_PER_GPU=1
    export GWP_NUM_WORKERS=0
    export GWP_MAX_STEPS=1
    export GWP_GRAD_ACCUM_STEPS=1
    export GWP_LOG_INTERVAL=1
    export GWP_WITH_EMA=0
    export GWP_DISABLE_CHECKPOINT=1
    export GWP_CONSEQUENCE_MODE=off
    export GWP_COUNTERFACTUAL_MASK_CACHE_PATHS="$SMOKE_CACHE"
    export GWP_COUNTERFACTUAL_MASK_WEIGHT=0.1
    "$POLICY_PYTHON" scripts/train.py --config giga_world_policy_0_5_libero_finetune
) > "$RESULT_ROOT/logs/smoke_train.log" 2>&1

status starting_policy_server
"$POLICY_PYTHON" scripts/inference_libero_gwp05.py \
    --checkpoint "$CHECKPOINT" \
    --vae "$VAE" \
    --norm-stats "$NORM_STATS" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --device "cuda:$RENDER_GPU" \
    --model-dtype bf16 \
    --num-inference-steps 10 \
    --num-views 2 \
    --seed 20260813 \
    > "$RESULT_ROOT/logs/policy_server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$RESULT_ROOT/policy_server.pid"

server_ready=0
for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        wait "$SERVER_PID"
    fi
    if grep -q 'Serving LIBERO GWP policy' "$RESULT_ROOT/logs/policy_server.log" 2>/dev/null; then
        server_ready=1
        break
    fi
    sleep 2
done
test "$server_ready" -eq 1

status exporting_train_actions
"$POLICY_PYTHON" scripts/evaluate_libero_open_loop.py \
    --dataset-root "$DATASET" \
    --suite libero_spatial \
    --episodes-file "$SPLIT_ROOT/train_episodes.txt" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --num-views 2 \
    --policy-checkpoint "$CHECKPOINT" \
    --policy-num-inference-steps 10 \
    --policy-model-dtype bf16 \
    --policy-seed 20260813 \
    --output-dir "$ACTIONS_TRAIN" \
    > "$RESULT_ROOT/logs/actions_train.log" 2>&1

status exporting_heldout_actions
"$POLICY_PYTHON" scripts/evaluate_libero_open_loop.py \
    --dataset-root "$DATASET" \
    --suite libero_spatial \
    --episodes-file "$SPLIT_ROOT/heldout_episodes.txt" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --num-views 2 \
    --policy-checkpoint "$CHECKPOINT" \
    --policy-num-inference-steps 10 \
    --policy-model-dtype bf16 \
    --policy-seed 20260813 \
    --output-dir "$ACTIONS_HELDOUT" \
    > "$RESULT_ROOT/logs/actions_heldout.log" 2>&1

kill -TERM "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

status rendering_train_masks
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$RENDER_GPU" "$EVAL_PYTHON" -B \
    scripts/generate_libero_counterfactual_mask_cache.py \
    --dataset-root "$DATASET" \
    --suite libero_spatial \
    --raw-demos-root "$RAW_DEMOS" \
    --predictions-root "$ACTIONS_TRAIN" \
    --policy-id "$CHECKPOINT" \
    --output-dir "$CACHE_TRAIN" \
    --episodes-file "$SPLIT_ROOT/train_episodes.txt" \
    --seed 20260813 \
    > "$RESULT_ROOT/logs/cache_train.log" 2>&1

status rendering_heldout_masks
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$RENDER_GPU" "$EVAL_PYTHON" -B \
    scripts/generate_libero_counterfactual_mask_cache.py \
    --dataset-root "$DATASET" \
    --suite libero_spatial \
    --raw-demos-root "$RAW_DEMOS" \
    --predictions-root "$ACTIONS_HELDOUT" \
    --policy-id "$CHECKPOINT" \
    --output-dir "$CACHE_HELDOUT" \
    --episodes-file "$SPLIT_ROOT/heldout_episodes.txt" \
    --seed 20260813 \
    > "$RESULT_ROOT/logs/cache_heldout.log" 2>&1

status running_preflight
"$EVAL_PYTHON" -B scripts/preflight_counterfactual_stage1.py \
    --project-root "$PROJECT" \
    --split-dir "$SPLIT_ROOT" \
    --train-cache "$CACHE_TRAIN" \
    --heldout-cache "$CACHE_HELDOUT" \
    --checkpoint "$CHECKPOINT" \
    --matched-output "$MATCHED_OUTPUT" \
    --full-output "$FULL_OUTPUT" \
    --report "$PREFLIGHT_REPORT" \
    > "$RESULT_ROOT/logs/preflight.log" 2>&1
"$POLICY_PYTHON" -B -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="passed"' \
    "$PREFLIGHT_REPORT"

status waiting_gpus_for_matched
wait_for_gpus
status training_matched
"$POLICY_PYTHON" -B scripts/launch_counterfactual_stage1_train.py \
    --preflight-report "$PREFLIGHT_REPORT" \
    --variant matched \
    --wan-pretrained "$WAN" \
    --norm-stats "$NORM_STATS" \
    > "$RESULT_ROOT/logs/matched_launcher.log" 2>&1

status waiting_gpus_for_full
wait_for_gpus
status training_full
"$POLICY_PYTHON" -B scripts/launch_counterfactual_stage1_train.py \
    --preflight-report "$PREFLIGHT_REPORT" \
    --variant full \
    --wan-pretrained "$WAN" \
    --norm-stats "$NORM_STATS" \
    > "$RESULT_ROOT/logs/full_launcher.log" 2>&1

MATCHED_CHECKPOINT=$(find "$MATCHED_OUTPUT/models" -type d \
    -path '*/checkpoint_*_step_200/transformer' -print -quit)
FULL_CHECKPOINT=$(find "$FULL_OUTPUT/models" -type d \
    -path '*/checkpoint_*_step_200/transformer' -print -quit)
test -n "$MATCHED_CHECKPOINT"
test -n "$FULL_CHECKPOINT"

status waiting_gpu_for_evaluation
wait_for_gpus
status evaluating_matched
CUDA_VISIBLE_DEVICES="$RENDER_GPU" "$POLICY_PYTHON" -B \
    scripts/evaluate_counterfactual_mask_head.py \
    --checkpoint "$MATCHED_CHECKPOINT" \
    --vae "$VAE" \
    --dataset "$DATASET" \
    --cache "$CACHE_HELDOUT" \
    --norm-stats "$NORM_STATS" \
    --output "$MATCHED_EVAL" \
    --mean-mask-cache "$CACHE_TRAIN" \
    --device cuda:0 \
    --model-dtype bf16 \
    --save-per-sample \
    > "$RESULT_ROOT/logs/matched_eval.log" 2>&1

status evaluating_full
CUDA_VISIBLE_DEVICES="$RENDER_GPU" "$POLICY_PYTHON" -B \
    scripts/evaluate_counterfactual_mask_head.py \
    --checkpoint "$FULL_CHECKPOINT" \
    --vae "$VAE" \
    --dataset "$DATASET" \
    --cache "$CACHE_HELDOUT" \
    --norm-stats "$NORM_STATS" \
    --output "$FULL_EVAL" \
    --mean-mask-cache "$CACHE_TRAIN" \
    --device cuda:0 \
    --model-dtype bf16 \
    --save-per-sample \
    > "$RESULT_ROOT/logs/full_eval.log" 2>&1

status auditing_pair
"$POLICY_PYTHON" -B scripts/audit_counterfactual_stage1_pair.py \
    --preflight-report "$PREFLIGHT_REPORT" \
    --matched-output "$MATCHED_OUTPUT" \
    --full-output "$FULL_OUTPUT" \
    --matched-eval "$MATCHED_EVAL" \
    --full-eval "$FULL_EVAL" \
    --output "$PAIR_AUDIT" \
    > "$RESULT_ROOT/logs/pair_audit.log" 2>&1

status complete
