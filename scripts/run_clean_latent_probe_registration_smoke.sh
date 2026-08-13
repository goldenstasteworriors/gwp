#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 8 ]; then
  echo "usage: $0 PROJECT_ROOT COMMIT COUNTERFACTUAL_CACHE OUTPUT_ROOT TRAIN_ENV EVAL_ENV GPU_ID WAIT_PID_CSV" >&2
  exit 2
fi

project_root=$1
target_commit=$2
counterfactual_cache=$3
output_root=$4
train_env=$5
eval_env=$6
gpu_id=$7
wait_pid_csv=$8

status_path="$output_root/status"
log_path="$output_root/pipeline.log"
worktree_path="${project_root}-probe-registration-${target_commit:0:12}"

mkdir -p "$output_root"
if [ -e "$status_path" ]; then
  echo "Refusing existing status: $status_path" >&2
  exit 1
fi
printf 'waiting_dependencies\n' > "$status_path"
exec > >(tee -a "$log_path") 2>&1

IFS=',' read -r -a wait_pids <<< "$wait_pid_csv"
for process_id in "${wait_pids[@]}"; do
  if [[ ! "$process_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid dependency PID: $process_id" >&2
    printf 'failed_invalid_dependency_pid\n' > "$status_path"
    exit 1
  fi
  while kill -0 "$process_id" 2>/dev/null; do
    sleep 30
  done
done

printf 'preparing_worktree\n' > "$status_path"
if [ -e "$worktree_path" ]; then
  echo "Refusing existing worktree path: $worktree_path" >&2
  printf 'failed_existing_worktree\n' > "$status_path"
  exit 1
fi
git -C "$project_root" worktree add --detach "$worktree_path" "$target_commit"

cleanup() {
  git -C "$project_root" worktree remove --force "$worktree_path" || true
}
trap cleanup EXIT

if [ ! -f "$counterfactual_cache/manifest.json" ]; then
  echo "Missing counterfactual smoke cache: $counterfactual_cache" >&2
  printf 'failed_missing_counterfactual_cache\n' > "$status_path"
  exit 1
fi
if [ ! -x "$train_env/bin/python" ] || [ ! -x "$eval_env/bin/python" ]; then
  echo "Missing requested conda environment" >&2
  printf 'failed_missing_environment\n' > "$status_path"
  exit 1
fi
if [[ ! "$gpu_id" =~ ^[2-5]$ ]]; then
  echo "Smoke GPU must be one of 2,3,4,5; got $gpu_id" >&2
  printf 'failed_gpu_policy\n' > "$status_path"
  exit 1
fi

demo_cache="$output_root/demo_mask_cache"
registration_report="$output_root/registration_report.json"
printf 'generating_demo_mask\n' > "$status_path"
(
  cd "$worktree_path"
  CUDA_VISIBLE_DEVICES="$gpu_id" MUJOCO_GL=egl "$eval_env/bin/python" \
    scripts/generate_libero_demonstration_mask_cache.py \
    --counterfactual-cache "$counterfactual_cache" \
    --output-dir "$demo_cache" \
    --seed 20260813
)

printf 'auditing_registration\n' > "$status_path"
(
  cd "$worktree_path"
  "$train_env/bin/python" scripts/audit_libero_demonstration_mask_registration.py \
    --cache "$demo_cache" \
    --report "$registration_report" \
    --max-shift-px 8 \
    --position-p95-mm 5 \
    --rotation-p95-deg 2 \
    --gripper-p95-max-abs 0.003 \
    --shift-p95-px 4 \
    --full-rgb-mae-p95 10 \
    --shifted-robot-rgb-mae-p95 15
)

registration_status=$("$train_env/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$registration_report")
if [ "$registration_status" = "passed" ]; then
  printf 'passed\n' > "$status_path"
else
  printf 'failed_registration_gate\n' > "$status_path"
fi
