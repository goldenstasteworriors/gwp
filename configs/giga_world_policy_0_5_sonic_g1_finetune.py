import os
import sys


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value


def _required(name: str) -> str:
    value = _env(name)
    if value is None:
        raise RuntimeError(f"{name} must be set for this training config")
    return value


def _int_list(name: str, default: str) -> list[int]:
    raw = _env(name, default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _episode_list(name: str) -> list[int] | None:
    raw = _env(name)
    if raw is None:
        return None
    episodes = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(value) for value in part.split("-", 1))
            if stop < start:
                raise ValueError(f"Invalid descending episode range {part!r}")
            episodes.extend(range(start, stop + 1))
        else:
            episodes.append(int(part))
    return sorted(set(episodes))


num_frames = int(_env("GWP_ACTION_HORIZON", "48"))
data_path = _required("GWP_SONIC_DATA_PATH")
norm_stats_path = _required("GWP_NORM_STATS_PATH")
transformer_pretrained = _required("GWP05_TRANSFORMER_PRETRAINED")
wan_pretrained = _env("GWP_WAN_PRETRAINED", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
project_dir = _env("GWP05_PROJECT_DIR", "outputs/gwp0_5_sonic_g1_finetune")
gpu_ids = [int(x) for x in _env("GWP_GPU_IDS", "0,1,2,3").split(",")]
physical_action_dim = int(_env("GWP_PHYSICAL_ACTION_DIM", "30"))
image_augmentation_enabled = _env("GWP_IMAGE_AUGMENTATION", "1") == "1"
view_keys = ["observation.images.ego_view"]
image_frame_offsets = _int_list("GWP_IMAGE_FRAME_OFFSETS", "0,12,24,36,48")
state_history_offsets = _int_list("GWP_STATE_HISTORY_OFFSETS", "0")
train_episodes = _episode_list("GWP_TRAIN_EPISODES")
max_ref_frames = int(_env("GWP_MAX_REF_FRAMES", "1"))
min_ref_latents = int(_env("GWP_MIN_REF_LATENTS", "1"))

delta_frames = {key: image_frame_offsets for key in view_keys}
if len(state_history_offsets) > 1 or state_history_offsets != [0]:
    delta_frames["observation.state"] = state_history_offsets

data_or_config = [
    dict(
        _class_name="WAMLeRobotDataset",
        data_path=data_path,
        data_size=None,
        delta_info={"action": num_frames},
        delta_frames=delta_frames,
        video_backend="pyav",
        t5_load_from=_env("GWP_T5_LOAD_FROM", "dir"),
        t5_cfg=dict(
            t5_embedding_dir=_env(
                "GWP_T5_EMBEDDING_DIR",
                os.path.join(data_path, "t5_embedding"),
            ),
            t5_embedding_pattern="episode_{episode_index:06d}.pt",
            t5_embedding_key="t5_embedding",
        ),
        t5_cache_size=4096,
        robotype="sonic_g1",
        episodes=train_episodes,
    )
]

config = dict(
    runners=["world_action_model.CasualWATrainerMoT"],
    project_dir=project_dir,
    launch=dict(
        gpu_ids=gpu_ids,
        distributed_type="DEEPSPEED",
        executable=_env(
            "GWP_ACCELERATE_EXECUTABLE",
            os.path.join(sys.prefix, "bin", "accelerate"),
        ),
        deepspeed_config=dict(
            deepspeed_config_file=_env(
                "GWP_DEEPSPEED_CONFIG",
                "accelerate_configs/zero2.json",
            )
        ),
        save_config_path=_env(
            "GWP_ACCELERATE_CONFIG_PATH",
            os.path.join(project_dir, "accelerate_config.json"),
        ),
        save_hostfile_path=_env(
            "GWP_ACCELERATE_HOSTFILE_PATH",
            os.path.join(project_dir, "accelerate_hostfile"),
        ),
        until_completion=False,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=int(_env("GWP_BATCH_SIZE_PER_GPU", "4")),
            num_workers=int(_env("GWP_NUM_WORKERS", "8")),
            pin_memory=True,
            transform=dict(
                type="WALeRobotTransformsPretrain",
                dst_size=(320, 384),
                num_frames=num_frames,
                is_train=True,
                norm_path=[{"path": norm_stats_path, "data_paths": [data_path]}],
                robotype_to_embodiment_id={"sonic_g1": 0},
                robotype_default_embodiment_id=0,
                model_action_dim=32,
                delta_mask_by_embodiment_id={"0": [False] * physical_action_dim},
                norm_use_quantiles=True,
                # This small dataset has near-constant 6D rotation components;
                # clamp quantile normalization so rare floating-point tails do
                # not amplify to 1e7 and dominate the action loss.
                norm_enable_clamp=True,
                num_views=1,
                view_keys=view_keys,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=max_ref_frames,
                        start=min_ref_latents,
                        factor=4,
                    )
                ),
                state_history_tokens=len(state_history_offsets),
                state_history_dropout_prob=float(
                    _env("GWP_STATE_HISTORY_DROPOUT_PROB", "0")
                ),
                append_episode_progress_token=(
                    _env("GWP_APPEND_EPISODE_PROGRESS_TOKEN", "0") == "1"
                ),
                append_state_velocity_token=(
                    _env("GWP_APPEND_STATE_VELOCITY_TOKEN", "0") == "1"
                ),
                emit_action_dim_mask=(
                    _env("GWP_MASK_PADDED_ACTION_DIMS", "0") == "1"
                ),
                emit_action_time_mask=(
                    _env("GWP_MASK_PADDED_ACTION_STEPS", "0") == "1"
                ),
                # Apply one sampled transform to every frame in a temporal chunk.
                # Hue and horizontal flips stay disabled because bottle color
                # and left/right arm identity are task semantics.
                image_augmentation=dict(
                    enabled=image_augmentation_enabled,
                    crop_scale=0.95,
                    rotation_degrees=5.0,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                ),
                max_prompt_len=64,
                subtask_prob=0,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=wan_pretrained,
        transformer_pretrained=transformer_pretrained,
        # The released checkpoint has 16-D action projectors. SONIC uses a
        # 30-D action padded to 32-D, so load shared weights and slice-copy the
        # pretrained projector weights into expanded 32-D projectors.
        strict_load=False,
        transformer=dict(
            added_kv_proj_dim=None,
            attention_head_dim=128,
            cross_attn_norm=True,
            eps=1e-6,
            ffn_dim=14336,
            freq_dim=256,
            image_dim=None,
            in_channels=48,
            num_attention_heads=24,
            num_layers=30,
            out_channels=48,
            patch_size=[1, 2, 2],
            pos_embed_seq_len=None,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=1024,
            text_dim=4096,
            action_expert_dim=1024,
            action_ffn_dim=4096,
            in_action_channels=32,
            out_action_channels=32,
            num_embodiments=2,
        ),
        visual_flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        visual_loss_weight=float(_env("GWP_VISUAL_LOSS_WEIGHT", "1")),
        action_loss_weight=float(_env("GWP_ACTION_LOSS_WEIGHT", "1")),
        action_temporal_weight_mode=_env(
            "GWP_ACTION_TEMPORAL_WEIGHT_MODE", "uniform"
        ),
        action_smoothness_weight=float(
            _env("GWP_ACTION_SMOOTHNESS_WEIGHT", "0")
        ),
        action_terminal_weight=float(
            _env("GWP_ACTION_TERMINAL_WEIGHT", "0")
        ),
        train_action_head_only=(
            _env("GWP_TRAIN_ACTION_HEAD_ONLY", "0") == "1"
        ),
        checkpoint_export_mode=_env("GWP_CHECKPOINT_EXPORT_MODE", "auto"),
        checkpoint_export_dtype=_env(
            "GWP_CHECKPOINT_EXPORT_DTYPE", "preserve"
        ),
        enable_gradient_checkpointing=_env("GWP_MODEL_GRADIENT_CHECKPOINTING", "0") == "1",
        skip_action_expert=False,
    ),
    optimizers=dict(type="CAME8Bit", lr=6e-5, weight_decay=1e-2),
    schedulers=dict(type="ConstantScheduler"),
    train=dict(
        resume=False,
        max_epochs=0,
        max_steps=int(_env("GWP_MAX_STEPS", "50000")),
        gradient_accumulation_steps=int(_env("GWP_GRAD_ACCUM_STEPS", "1")),
        mixed_precision="bf16",
        checkpoint_interval=int(_env("GWP_CHECKPOINT_INTERVAL", "5000")),
        checkpoint_total_limit=int(_env("GWP_CHECKPOINT_TOTAL_LIMIT", "-1")),
        checkpoint_weights_only=_env("GWP_CHECKPOINT_WEIGHTS_ONLY", "0") == "1",
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        # Keep logging disabled by default; W&B can be enabled per run without
        # re-enabling the incompatible PPU TensorBoard path.
        log_with=_env("GWP_LOG_WITH"),
        log_interval=int(_env("GWP_LOG_INTERVAL", "20")),
        with_ema=_env("GWP_WITH_EMA", "1") == "1",
        activation_checkpointing=_env("GWP_ACTIVATION_CHECKPOINTING", "0") == "1",
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
