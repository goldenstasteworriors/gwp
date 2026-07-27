import os
import sys


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value


def _required(name: str) -> str:
    value = _env(name)
    if value is None:
        raise RuntimeError(f"{name} must be set for HOT3D training")
    return value


index_path = _required("GWP_HOT3D_INDEX")
norm_stats_path = _required("GWP_NORM_STATS_PATH")
transformer_pretrained = _required("GWP05_TRANSFORMER_PRETRAINED")
wan_pretrained = _env("GWP_WAN_PRETRAINED", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
project_dir = _env("GWP05_PROJECT_DIR", "outputs/gwp0_5_hot3d_pretrain")
gpu_ids = [int(x) for x in _env("GWP_GPU_IDS", "4,5,6,7").split(",")]

config = dict(
    runners=["world_action_model.CasualWATrainerMoT"],
    project_dir=project_dir,
    launch=dict(
        gpu_ids=gpu_ids,
        distributed_type="DEEPSPEED",
        executable=_env("GWP_ACCELERATE_EXECUTABLE", os.path.join(sys.prefix, "bin", "accelerate")),
        deepspeed_config=dict(
            deepspeed_config_file=_env("GWP_DEEPSPEED_CONFIG", "accelerate_configs/zero2.json")
        ),
        until_completion=False,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=dict(
                _class_name="HOT3DWAMDataset",
                index_path=index_path,
                action_horizon=48,
                image_frame_offsets=[0, 12, 24, 36, 48],
                robotype="hot3d_human",
                max_prompt_len=64,
            ),
            batch_size_per_gpu=int(_env("GWP_BATCH_SIZE_PER_GPU", "1")),
            num_workers=int(_env("GWP_NUM_WORKERS", "4")),
            pin_memory=True,
            transform=dict(
                type="WAHot3DTransforms",
                norm_path=norm_stats_path,
                dst_size=(320, 384),
                fps=30,
                embodiment_id=0,
                image_cfg=dict(mask_generator=dict(max_ref_frames=1, start=1, factor=4)),
                max_prompt_len=64,
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
            in_action_channels=53,
            out_action_channels=53,
            num_embodiments=2,
        ),
        visual_flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        enable_gradient_checkpointing=True,
        skip_action_expert=False,
    ),
    optimizers=dict(type="CAME8Bit", lr=float(_env("GWP_LEARNING_RATE", "1e-5")), weight_decay=1e-2),
    schedulers=dict(type="ConstantScheduler"),
    train=dict(
        resume=False,
        max_epochs=0,
        max_steps=int(_env("GWP_MAX_STEPS", "50000")),
        gradient_accumulation_steps=int(_env("GWP_GRAD_ACCUM_STEPS", "1")),
        mixed_precision="bf16",
        checkpoint_interval=int(_env("GWP_CHECKPOINT_INTERVAL", "1000")),
        checkpoint_total_limit=int(_env("GWP_CHECKPOINT_TOTAL_LIMIT", "5")),
        checkpoint_weights_only=False,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with="wandb",
        log_interval=int(_env("GWP_LOG_INTERVAL", "10")),
        with_ema=False,
        activation_checkpointing=False,
    ),
    test=dict(),
)
