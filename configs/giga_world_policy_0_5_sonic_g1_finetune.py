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
data_path = _required("GWP_SONIC_DATA_PATH")
norm_stats_path = _required("GWP_NORM_STATS_PATH")
transformer_pretrained = _required("GWP05_TRANSFORMER_PRETRAINED")
wan_pretrained = _env("GWP_WAN_PRETRAINED", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
project_dir = _env("GWP05_PROJECT_DIR", "outputs/gwp0_5_sonic_g1_finetune")
gpu_ids = [int(x) for x in _env("GWP_GPU_IDS", "0,1,2,3").split(",")]
view_keys = ["observation.images.ego_view"]
image_frame_offsets = [0, 12, 24, 36, 48]

data_or_config = [
    dict(
        _class_name="WAMLeRobotDataset",
        data_path=data_path,
        data_size=None,
        delta_info={"action": num_frames},
        delta_frames={key: image_frame_offsets for key in view_keys},
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
                delta_mask_by_embodiment_id={"0": [False] * 30},
                norm_use_quantiles=True,
                norm_enable_clamp=False,
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
        enable_gradient_checkpointing=False,
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
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        # tensorboardX on the PPU image wraps scalar metrics as ndarrays and
        # raises during add_scalar; console logging remains enabled.
        log_with=None,
        log_interval=int(_env("GWP_LOG_INTERVAL", "20")),
        with_ema=True,
        activation_checkpointing=False,
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
