"""GWP-0.5 contact-transition pretraining on processed EgoHumanoid data."""

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


num_frames = 48
data_path = _required("GWP_EGO_DATA_PATH")
norm_stats_path = _required("GWP_EGO_NORM_STATS_PATH")
contact_sidecar_dir = _required("GWP_CONTACT_SIDECAR_DIR")
transformer_pretrained = _required("GWP05_TRANSFORMER_PRETRAINED")
wan_pretrained = _env("GWP_WAN_PRETRAINED", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
project_dir = _env("GWP05_PROJECT_DIR", "outputs/gwp0_5_ego_contact_pretrain")
gpu_ids = [int(value) for value in _env("GWP_GPU_IDS", "0,1,2,3").split(",")]
view_keys = ["observation.images.left"]
image_frame_offsets = [0, 12, 24, 36, 48]
episode_text = _env("GWP_EGO_EPISODES", "")
episodes = [int(value) for value in episode_text.split(",") if value.strip()]

dataset_config = dict(
    _class_name="WAMLeRobotDataset",
    data_path=data_path,
    data_size=None,
    delta_frames={
        "action_delta_eef": list(range(num_frames + 1)),
        "hand_status": list(range(num_frames + 1)),
        **{key: image_frame_offsets for key in view_keys},
    },
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
    contact_sidecar_dir=contact_sidecar_dir,
    contact_sidecar_pattern="episode_{episode_index:06d}.npz",
    contact_sidecar_cache_size=int(_env("GWP_CONTACT_SIDECAR_CACHE_SIZE", "8")),
    contact_sidecar_strict=True,
    robotype="ego_human",
)
if episodes:
    dataset_config["episodes"] = episodes

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
        until_completion=False,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[dataset_config],
            batch_size_per_gpu=int(_env("GWP_BATCH_SIZE_PER_GPU", "4")),
            num_workers=int(_env("GWP_NUM_WORKERS", "8")),
            pin_memory=True,
            transform=dict(
                type="WALeRobotTransformsPretrain",
                dst_size=(320, 384),
                num_frames=num_frames,
                is_train=True,
                norm_path=[{"path": norm_stats_path, "data_paths": [data_path]}],
                robotype_to_embodiment_id={"sonic_g1": 0, "ego_human": 1},
                robotype_default_embodiment_id=1,
                model_action_dim=32,
                delta_mask_by_embodiment_id={"0": [False] * 25, "1": [False] * 20},
                norm_use_quantiles=True,
                norm_enable_clamp=True,
                num_views=1,
                view_keys=view_keys,
                ego_dual_eef_cfg=dict(
                    pose_key="action_eef",
                    delta_key="action_delta_eef",
                    hand_status_key="hand_status",
                    # delta[i] is pose[i-1] -> pose[i]. Keep frame 0 only to
                    # recover current grip state, then predict frames 1..48.
                    chunk_start_offset=1,
                ),
                image_cfg=dict(
                    mask_generator=dict(max_ref_frames=1, start=1, factor=4)
                ),
                image_augmentation=dict(
                    enabled=True,
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
            num_contact_tokens=int(_env("GWP_CONTACT_TOKENS", "12")),
            contact_target_dim=54,
            contact_condition_dim=54,
            contact_bottleneck=_env("GWP_CONTACT_STRICT_BOTTLENECK", "1") == "1",
        ),
        visual_flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        enable_gradient_checkpointing=_env("GWP_MODEL_GRADIENT_CHECKPOINTING", "0") == "1",
        skip_action_expert=False,
        contact_loss_weight=float(_env("GWP_CONTACT_LOSS_WEIGHT", "0.25")),
        contact_loss_type=_env("GWP_CONTACT_LOSS_TYPE", "bce"),
        contact_condition_mode=_env("GWP_CONTACT_CONDITION_MODE", "predicted"),
        require_contact_targets=_env("GWP_REQUIRE_CONTACT_TARGETS", "1") == "1",
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
        log_with=None,
        log_interval=int(_env("GWP_LOG_INTERVAL", "20")),
        with_ema=_env("GWP_WITH_EMA", "1") == "1",
        activation_checkpointing=_env("GWP_ACTIVATION_CHECKPOINTING", "0") == "1",
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
