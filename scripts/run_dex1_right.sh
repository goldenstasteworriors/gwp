#!/usr/bin/env bash
set -euo pipefail
variant="${1:?Usage: bash scripts/run_dex1_right.sh all|episode0|episode0_mem [--dry-run]}"
mode="${2:-train}"
case "$variant" in
  all) gpu=0; dataset=all; memory=0 ;;
  episode0) gpu=1; dataset=episode0; memory=0 ;;
  episode0_mem) gpu=2; dataset=episode0; memory=1 ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
if [[ "$mode" != train && "$mode" != --dry-run ]]; then
  echo "Unknown mode: $mode" >&2; exit 2
fi
project_root="${GWP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$project_root"
python_bin="${GWP_PYTHON:-/nix/conda_envs/gigaworld-policy/bin/python}"
data_root="${GWP_DEX1_DATA_ROOT:-$project_root/data/dex1_right_measured_20261021}"
asset_root="${GWP_ASSET_ROOT:-$project_root/../giga-world-policy-assets}"
runtime_cache="${GWP_RUNTIME_CACHE:-/nix/gigaworld-policy-cache}"
run_date="$(TZ=CST-8 date +%Y%m%d_%H%M%S)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export GWP_GPU_IDS=0
export GWP_PHYSICAL_ACTION_DIM=8
export GWP_SHORT_MEMORY="$memory"
export GWP_MODEL_GRADIENT_CHECKPOINTING="${GWP_MODEL_GRADIENT_CHECKPOINTING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GWP_LOG_INTERVAL="${GWP_LOG_INTERVAL:-1}"
export GWP_SONIC_DATA_PATH="$data_root/$dataset"
export GWP_NORM_STATS_PATH="$GWP_SONIC_DATA_PATH/norm.json"
export GWP_T5_LOAD_FROM=dir
export GWP_T5_EMBEDDING_DIR="$GWP_SONIC_DATA_PATH/t5_embedding"
export GWP_WAN_PRETRAINED="${GWP_WAN_PRETRAINED:-$asset_root/Wan2.2-TI2V-5B-Diffusers}"
export GWP05_TRANSFORMER_PRETRAINED="${GWP05_TRANSFORMER_PRETRAINED:-$asset_root/Giga-World-Policy-0.5}"
export GWP05_PROJECT_DIR="${GWP05_PROJECT_DIR:-$project_root/outputs/dex1_right_${variant}_${run_date}}"
export TMPDIR="$runtime_cache/tmp"
export HF_HOME="$runtime_cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$runtime_cache/xdg"
export TORCH_HOME="$runtime_cache/torch"
export TORCH_EXTENSIONS_DIR="$runtime_cache/torch_extensions"
export PYTHONUNBUFFERED=1
if [[ "$mode" == --dry-run ]]; then
  "$python_bin" -c 'import json; from configs.giga_world_policy_0_5_dex1_right_measured import config; print(json.dumps(config,indent=2))'
  exit
fi
for path in "$python_bin" "$GWP_NORM_STATS_PATH" "$GWP_T5_EMBEDDING_DIR/episode_000000.pt" "$GWP_WAN_PRETRAINED/vae/config.json"; do
  [[ -f "$path" ]] || { echo "Required file missing: $path" >&2; exit 1; }
done
[[ -d "$GWP05_TRANSFORMER_PRETRAINED" ]] || { echo "Missing GWP transformer weights" >&2; exit 1; }
if [[ -n "${GWP_RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "$GWP_RESUME_CHECKPOINT/custom_checkpoint_0.pkl" && -f "$GWP_RESUME_CHECKPOINT/optimizer.bin" ]] || { echo "Incomplete resume checkpoint: $GWP_RESUME_CHECKPOINT" >&2; exit 1; }
fi
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$GWP05_PROJECT_DIR"
"$python_bin" -c 'import json,os; from configs.giga_world_policy_0_5_dex1_right_measured import config; p=os.path.join(os.environ["GWP05_PROJECT_DIR"],"launch_config.json"); json.dump(config,open(p,"x"),indent=2)'
echo "variant=$variant physical_gpu=$gpu output=$GWP05_PROJECT_DIR"
"$python_bin" scripts/train.py --config giga_world_policy_0_5_dex1_right_measured 2>&1 | tee "$GWP05_PROJECT_DIR/console.log"
