#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/user/ykj/project/giga-world-policy-camvla
ASSETS=/data/ykj_gigaworld_policy
ENV_PREFIX="$ASSETS/envs/gigaworld-policy"
DATA="$ASSETS/datasets/pipette_camera_delta_nominal"
WAN="$ASSETS/models/wan_vae"
INITIAL="$ASSETS/models/gwp05_sonic_init/transformer"

for required in \
  "$ENV_PREFIX/bin/python" \
  "$DATA/meta/info.json" \
  "$DATA/norm_stats.json" \
  "$DATA/t5_embedding/episode_000000.pt" \
  "$WAN/vae/config.json" \
  "$WAN/vae/diffusion_pytorch_model.safetensors" \
  "$INITIAL/diffusion_pytorch_model.bin"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required asset: $required" >&2
    exit 2
  fi
done

export HF_HOME="$ASSETS/cache/hf"
export TMPDIR="$ASSETS/cache/tmp"
export PIP_CACHE_DIR="$ASSETS/cache/pip"
export GWP_SONIC_DATA_PATH="$DATA"
export GWP_NORM_STATS_PATH="$DATA/norm_stats.json"
export GWP_T5_LOAD_FROM=dir
export GWP_T5_EMBEDDING_DIR="$DATA/t5_embedding"
export GWP_WAN_PRETRAINED="$WAN"
export GWP05_TRANSFORMER_PRETRAINED="$INITIAL"
export GWP_GPU_IDS="${GWP_GPU_IDS:-0}"
export GWP_BATCH_SIZE_PER_GPU="${GWP_BATCH_SIZE_PER_GPU:-1}"
export GWP_NUM_WORKERS="${GWP_NUM_WORKERS:-0}"
export GWP_MAX_STEPS="${GWP_MAX_STEPS:-2}"
export GWP_CHECKPOINT_INTERVAL="${GWP_CHECKPOINT_INTERVAL:-2}"
export GWP_LOG_INTERVAL="${GWP_LOG_INTERVAL:-1}"
export GWP_WITH_EMA=0
export GWP05_PROJECT_DIR="${GWP05_PROJECT_DIR:-$ASSETS/runs/camvla_wam_$(date +%F_%H%M%S)}"

mkdir -p "$GWP05_PROJECT_DIR"
cd "$PROJECT"
exec "$ENV_PREFIX/bin/python" scripts/train.py \
  --config giga_world_policy_0_5_sonic_g1_camvla_wam
