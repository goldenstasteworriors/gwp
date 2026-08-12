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
    return [int(item.strip()) for item in _env(name, default).split(",") if item.strip()]


def _bool(name: str, default: bool) -> bool:
    value = _env(name, "1" if default else "0").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"{name} must be a boolean value")
    return value in {"1", "true", "yes"}


num_frames = 48
data_path = _required("GWP_LIBERO_DATA_PATH")
norm_stats_path = _required("GWP_NORM_STATS_PATH")
transformer_pretrained = _required("GWP05_TRANSFORMER_PRETRAINED")
wan_pretrained = _env("GWP_WAN_PRETRAINED", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
project_dir = _env("GWP05_PROJECT_DIR", "outputs/gwp0_5_libero_finetune")
gpu_ids = _int_list("GWP_GPU_IDS", "0,1,2,3,4,5,6,7")

consequence_mode = _env("GWP_CONSEQUENCE_MODE", "off")
if consequence_mode not in {"off", "policy_only", "joint", "video_only"}:
    raise ValueError(
        "GWP_CONSEQUENCE_MODE must be one of off, policy_only, joint, video_only"
    )
consequence_weight = 0.0
if consequence_mode != "off":
    consequence_weight = float(_env("GWP_CONSEQUENCE_WEIGHT", "0.1"))

view_keys = ["observation.images.image"]
image_frame_offsets = [0, 12, 24, 36, 48]

data_or_config = [
    dict(
        _class_name="WAMLeRobotDataset",
        data_path=data_path,
        data_size=None,
        delta_info={"action": num_frames},
        delta_frames={key: image_frame_offsets for key in view_keys},
        video_backend="pyav",
        t5_load_from="dir",
        t5_cfg=dict(
            t5_embedding_dir=_env(
                "GWP_T5_EMBEDDING_DIR",
                os.path.join(data_path, "t5_embedding"),
            ),
            t5_embedding_pattern="episode_{episode_index:06d}.pt",
            t5_embedding_key="t5_embedding",
        ),
        t5_cache_size=4096,
        robotype="franka",
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
        deepspeed_config=dict(deepspeed_config_file="accelerate_configs/zero2.json"),
        until_completion=False,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=int(_env("GWP_BATCH_SIZE_PER_GPU", "2")),
            num_workers=int(_env("GWP_NUM_WORKERS", "8")),
            pin_memory=True,
            transform=dict(
                type="WALeRobotTransformsPretrain",
                dst_size=(256, 256),
                num_frames=num_frames,
                fps=20,
                is_train=True,
                norm_path=[{"path": norm_stats_path, "data_paths": [data_path]}],
                robotype_to_embodiment_id={"franka": 0},
                robotype_default_embodiment_id=0,
                model_action_dim=16,
                # LIBERO actions are already relative EEF controller commands;
                # subtracting the absolute EEF state would be dimensionally wrong.
                delta_mask_by_embodiment_id={"0": [False] * 7},
                norm_use_quantiles=True,
                norm_enable_clamp=True,
                num_views=1,
                view_keys=view_keys,
                image_cfg=dict(
                    mask_generator=dict(max_ref_frames=1, start=1, factor=4)
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
        strict_load=True,
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
            in_action_channels=16,
            out_action_channels=16,
            num_embodiments=2,
        ),
        visual_flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        enable_gradient_checkpointing=True,
        skip_action_expert=False,
        consequence_loss_weight=consequence_weight,
        consequence_gradient_mode=(
            "policy_only" if consequence_mode == "off" else consequence_mode
        ),
        consequence_interval=int(_env("GWP_CONSEQUENCE_INTERVAL", "4")),
        consequence_start_step=int(_env("GWP_CONSEQUENCE_START_STEP", "0")),
        consequence_visual_sigma=float(_env("GWP_CONSEQUENCE_VISUAL_SIGMA", "1.0")),
    ),
    optimizers=dict(type="CAME8Bit", lr=6e-5, weight_decay=1e-2),
    schedulers=dict(type="ConstantScheduler"),
    train=dict(
        resume=False,
        max_epochs=0,
        max_steps=int(_env("GWP_MAX_STEPS", "6000")),
        gradient_accumulation_steps=int(_env("GWP_GRAD_ACCUM_STEPS", "1")),
        mixed_precision="bf16",
        checkpoint_interval=int(_env("GWP_CHECKPOINT_INTERVAL", "1000")),
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with=None,
        log_interval=int(_env("GWP_LOG_INTERVAL", "20")),
        with_ema=_bool("GWP_WITH_EMA", True),
        activation_checkpointing=True,
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
