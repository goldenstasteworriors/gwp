#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 12 ]; then
    echo "usage: $0 PROJECT_ROOT COMMIT STAGE_ROOT OUTPUT_ROOT TRAIN_ENV EVAL_ENV GPU_ID WAIT_PID BASELINE_CHECKPOINT VAE NORM_STATS SPLIT_ROOT" >&2
    exit 2
fi

project_root=$1
target_commit=$2
stage_root=$3
output_root=$4
train_env=$5
eval_env=$6
gpu_id=$7
wait_pid=$8
baseline_checkpoint=$9
vae=${10}
norm_stats=${11}
split_root=${12}

status_path="$output_root/status"
log_path="$output_root/pipeline.log"
worktree_path="${project_root}-clean-probe-gate-${target_commit:0:12}"
train_counterfactual_cache="$stage_root/cache_train"
heldout_counterfactual_cache="$stage_root/cache_heldout"
train_demo_cache="$output_root/demo_mask_train"
heldout_demo_cache="$output_root/demo_mask_heldout"
train_registration="$output_root/registration_train.json"
heldout_registration="$output_root/registration_heldout.json"
probe_output="$output_root/probe_training"
probe_report="$probe_output/report.json"
probe_checkpoint="$probe_output/probe"
action_gap_report="$output_root/action_gap.json"

status() {
    printf '%s\n' "$1" > "$status_path"
}

cleanup() {
    local code=$?
    if [ -e "$worktree_path" ]; then
        git -C "$project_root" worktree remove --force "$worktree_path" || true
    fi
    if [ "$code" -ne 0 ]; then
        cat "$status_path" > "$output_root/failed_stage" 2>/dev/null || true
        status failed
    fi
    printf '%s\n' "$code" > "$output_root/exit_code"
}

wait_for_gpu() {
    while nvidia-smi -i "$gpu_id" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
        sleep 60
    done
}

mkdir -p "$output_root"
if [ -e "$status_path" ]; then
    echo "Refusing existing status: $status_path" >&2
    exit 1
fi
status waiting_stage1_pipeline
exec > >(tee -a "$log_path") 2>&1
trap cleanup EXIT

if [[ ! "$wait_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid Stage-1 dependency PID: $wait_pid" >&2
    exit 2
fi
while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 60
done

status validating_dependencies
test -f "$train_counterfactual_cache/manifest.json"
test -f "$heldout_counterfactual_cache/manifest.json"
test -d "$baseline_checkpoint"
test -d "$vae"
test -f "$norm_stats"
test -f "$split_root/train_episodes.txt"
test -f "$split_root/heldout_episodes.txt"
test -x "$train_env/bin/python"
test -x "$eval_env/bin/python"
train_dataset=$(
    "$train_env/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["dataset_root"])' \
        "$train_counterfactual_cache/manifest.json"
)
dataset=$(
    "$train_env/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["dataset_root"])' \
        "$heldout_counterfactual_cache/manifest.json"
)
test "$(realpath "$train_dataset")" = "$(realpath "$dataset")"
test -d "$dataset"
output_device=$(stat -c '%d' "$output_root")
for source in "$stage_root" "$baseline_checkpoint" "$vae" "$dataset"; do
    if [ "$(stat -c '%d' "$source")" != "$output_device" ]; then
        echo "Probe gate source/output crosses filesystems: $source" >&2
        exit 1
    fi
done
if [[ ! "$gpu_id" =~ ^[2-5]$ ]]; then
    echo "Probe GPU must be one of 2,3,4,5; got $gpu_id" >&2
    exit 2
fi
for path in "$train_demo_cache" "$heldout_demo_cache" "$train_registration" \
    "$heldout_registration" "$probe_output" "$action_gap_report"; do
    test ! -e "$path"
done

status preparing_worktree
if [ -e "$worktree_path" ]; then
    echo "Refusing existing worktree path: $worktree_path" >&2
    exit 1
fi
git -C "$project_root" worktree add --detach "$worktree_path" "$target_commit"
git -C "$worktree_path" rev-parse HEAD > "$output_root/code_commit.txt"
test "$(git -C "$worktree_path" rev-parse HEAD)" = "$target_commit"
test -z "$(git -C "$worktree_path" status --porcelain --untracked-files=no)"

status waiting_gpu_for_demo_replay
wait_for_gpu
status generating_train_demo_masks
(
    cd "$worktree_path"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$eval_env/bin/python" -B \
        scripts/generate_libero_demonstration_mask_cache.py \
        --counterfactual-cache "$train_counterfactual_cache" \
        --output-dir "$train_demo_cache" \
        --seed 20260813
)

status generating_heldout_demo_masks
(
    cd "$worktree_path"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$eval_env/bin/python" -B \
        scripts/generate_libero_demonstration_mask_cache.py \
        --counterfactual-cache "$heldout_counterfactual_cache" \
        --output-dir "$heldout_demo_cache" \
        --seed 20260813
)

status auditing_train_registration
(
    cd "$worktree_path"
    "$train_env/bin/python" -B scripts/audit_libero_demonstration_mask_registration.py \
        --cache "$train_demo_cache" \
        --report "$train_registration" \
        --max-shift-px 8 \
        --position-p95-mm 5 \
        --rotation-p95-deg 2 \
        --gripper-p95-max-abs 0.003 \
        --shift-p95-px 4 \
        --full-rgb-mae-p95 10 \
        --shifted-robot-rgb-mae-p95 15
)

status auditing_heldout_registration
(
    cd "$worktree_path"
    "$train_env/bin/python" -B scripts/audit_libero_demonstration_mask_registration.py \
        --cache "$heldout_demo_cache" \
        --report "$heldout_registration" \
        --max-shift-px 8 \
        --position-p95-mm 5 \
        --rotation-p95-deg 2 \
        --gripper-p95-max-abs 0.003 \
        --shift-p95-px 4 \
        --full-rgb-mae-p95 10 \
        --shifted-robot-rgb-mae-p95 15
)

registration_status=$(
    "$train_env/bin/python" -c \
        'import json,sys; print("passed" if all(json.load(open(path))["status"] == "passed" for path in sys.argv[1:]) else "failed")' \
        "$train_registration" "$heldout_registration"
)
if [ "$registration_status" != "passed" ]; then
    status stopped_registration_gate_failed
    exit 0
fi

status waiting_gpu_for_probe_training
wait_for_gpu
status training_and_gating_probe
(
    cd "$worktree_path"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$train_env/bin/python" -B \
        scripts/train_libero_latent_mask_probe.py \
        --train-cache "$train_demo_cache" \
        --heldout-cache "$heldout_demo_cache" \
        --train-registration "$train_registration" \
        --heldout-registration "$heldout_registration" \
        --vae "$vae" \
        --output-dir "$probe_output" \
        --device cuda:0 \
        --batch-size 4 \
        --epochs 20 \
        --learning-rate 0.001 \
        --weight-decay 0.0001 \
        --seed 20260813 \
        --threshold 0.5 \
        --num-workers 2
)

probe_status=$(
    "$train_env/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$probe_report"
)
if [ "$probe_status" != "passed" ]; then
    status stopped_probe_gate_failed
    exit 0
fi

status waiting_gpu_for_action_gap
wait_for_gpu
status evaluating_action_conditioning_gap
(
    cd "$worktree_path"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$train_env/bin/python" -B \
        scripts/evaluate_clean_latent_action_gap.py \
        --checkpoint "$baseline_checkpoint" \
        --vae "$vae" \
        --dataset "$dataset" \
        --cache "$heldout_counterfactual_cache" \
        --demonstration-cache "$heldout_demo_cache" \
        --demonstration-registration "$heldout_registration" \
        --norm-stats "$norm_stats" \
        --mask-probe "$probe_checkpoint" \
        --output "$action_gap_report" \
        --device cuda:0 \
        --model-dtype bf16 \
        --seed 20260813 \
        --threshold 0.5 \
        --bootstrap-samples 10000
)

gap_status=$(
    "$train_env/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$action_gap_report"
)
if [ "$gap_status" != "passed" ]; then
    status stopped_action_gap_failed
    exit 0
fi

status ready_clean_latent_training
