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
data_paths_value = _env("GWP_LIBERO_DATA_PATHS")
if data_paths_value is None:
    data_paths_value = _required("GWP_LIBERO_DATA_PATH")
data_paths = [path.strip() for path in data_paths_value.split(",") if path.strip()]
if not data_paths:
    raise RuntimeError("GWP_LIBERO_DATA_PATH or GWP_LIBERO_DATA_PATHS must contain a dataset path")
norm_stats_path = _required("GWP_NORM_STATS_PATH")
transformer_pretrained = _required("GWP05_TRANSFORMER_PRETRAINED")
wan_pretrained = _env("GWP_WAN_PRETRAINED", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
project_dir = _env("GWP05_PROJECT_DIR", "outputs/gwp0_5_libero_finetune")
gpu_ids = _int_list("GWP_GPU_IDS", "0,1,2,3,4,5,6,7")
checkpoint_model_dtype = _env("GWP_CHECKPOINT_MODEL_DTYPE", "original").lower()
if checkpoint_model_dtype not in {"original", "bf16", "fp16"}:
    raise ValueError(
        "GWP_CHECKPOINT_MODEL_DTYPE must be one of original, bf16, fp16"
    )
with_ema = _bool("GWP_WITH_EMA", True)
if checkpoint_model_dtype != "original" and with_ema:
    raise ValueError(
        "Compact checkpoint casting requires GWP_WITH_EMA=0; EMA uses a separate state dict"
    )

consequence_mode = _env("GWP_CONSEQUENCE_MODE", "off")
if consequence_mode not in {"off", "policy_only", "joint", "video_only"}:
    raise ValueError(
        "GWP_CONSEQUENCE_MODE must be one of off, policy_only, joint, video_only"
    )
consequence_weight = 0.0
if consequence_mode != "off":
    consequence_weight = float(_env("GWP_CONSEQUENCE_WEIGHT", "0.1"))

counterfactual_mask_cache_value = _env("GWP_COUNTERFACTUAL_MASK_CACHE_PATHS")
counterfactual_mask_cache_paths = []
if counterfactual_mask_cache_value is not None:
    counterfactual_mask_cache_paths = [
        path.strip() for path in counterfactual_mask_cache_value.split(",") if path.strip()
    ]
    if len(counterfactual_mask_cache_paths) != len(data_paths):
        raise ValueError(
            "GWP_COUNTERFACTUAL_MASK_CACHE_PATHS must contain one path per LIBERO dataset"
        )
    if consequence_mode != "off":
        raise ValueError(
            "Counterfactual mask supervision cannot use the legacy demo-RGB consequence loss"
        )
counterfactual_mask_weight = (
    float(_env("GWP_COUNTERFACTUAL_MASK_WEIGHT", "0.1"))
    if counterfactual_mask_cache_paths
    else 0.0
)
counterfactual_mask_compute_control = _bool(
    "GWP_COUNTERFACTUAL_MASK_COMPUTE_CONTROL", False
)
if counterfactual_mask_compute_control:
    if not counterfactual_mask_cache_paths:
        raise ValueError("Mask compute control requires a counterfactual mask cache")
    if counterfactual_mask_weight != 0.0:
        raise ValueError("Mask compute control is only valid when mask weight is zero")
counterfactual_mask_trainable_scope = _env(
    "GWP_COUNTERFACTUAL_MASK_TRAINABLE_SCOPE", "all"
)
counterfactual_mask_alignment_mode = _env(
    "GWP_COUNTERFACTUAL_MASK_ALIGNMENT_MODE", "hidden_head"
)
if counterfactual_mask_alignment_mode not in {
    "hidden_head",
    "clean_latent_probe",
    "shared_mask_video_flow",
}:
    raise ValueError(
        "GWP_COUNTERFACTUAL_MASK_ALIGNMENT_MODE must be hidden_head, "
        "clean_latent_probe, or shared_mask_video_flow"
    )
counterfactual_mask_probe_pretrained = _env(
    "GWP_COUNTERFACTUAL_MASK_PROBE_PRETRAINED"
)
counterfactual_action_rank_weight = float(
    _env("GWP_COUNTERFACTUAL_ACTION_RANK_WEIGHT", "0.0")
)
counterfactual_action_rank_compute_control = _bool(
    "GWP_COUNTERFACTUAL_ACTION_RANK_COMPUTE_CONTROL", False
)
counterfactual_action_rank_margin = float(
    _env("GWP_COUNTERFACTUAL_ACTION_RANK_MARGIN", "0.01")
)
counterfactual_action_rank_min_mse = float(
    _env("GWP_COUNTERFACTUAL_ACTION_RANK_MIN_MSE", "0.0001")
)
if (
    counterfactual_action_rank_weight > 0.0
    or counterfactual_action_rank_compute_control
):
    if counterfactual_mask_alignment_mode != "shared_mask_video_flow":
        raise ValueError("Action ranking requires shared_mask_video_flow")
    if counterfactual_mask_weight <= 0.0:
        raise ValueError("Action ranking requires positive mask-flow supervision")
if counterfactual_mask_trainable_scope not in {"all", "head_only"}:
    raise ValueError(
        "GWP_COUNTERFACTUAL_MASK_TRAINABLE_SCOPE must be one of all, head_only"
    )
if counterfactual_mask_trainable_scope == "head_only" and counterfactual_mask_weight <= 0.0:
    raise ValueError("head_only requires positive counterfactual mask supervision")

view_keys = [
    key.strip()
    for key in _env("GWP_VIEW_KEYS", "observation.images.image").split(",")
    if key.strip()
]
if not view_keys:
    raise ValueError("GWP_VIEW_KEYS must contain at least one image feature")
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
        counterfactual_mask_cache=(
            counterfactual_mask_cache_paths[index]
            if counterfactual_mask_cache_paths
            else None
        ),
    )
    for index, data_path in enumerate(data_paths)
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
                norm_path=[{"path": norm_stats_path, "data_paths": data_paths}],
                robotype_to_embodiment_id={"franka": 0},
                robotype_default_embodiment_id=0,
                model_action_dim=16,
                # LIBERO actions are already relative EEF controller commands;
                # subtracting the absolute EEF state would be dimensionally wrong.
                delta_mask_by_embodiment_id={"0": [False] * 7},
                norm_use_quantiles=True,
                norm_enable_clamp=True,
                num_views=len(view_keys),
                view_keys=view_keys,
                view_concat_mode=_env("GWP_VIEW_CONCAT_MODE", "horizontal"),
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
            counterfactual_mask_frames_per_latent=(
                4
                if counterfactual_mask_cache_paths
                and counterfactual_mask_alignment_mode == "hidden_head"
                else 0
            ),
            use_mask_video_modality_embedding=(
                bool(counterfactual_mask_cache_paths)
                and counterfactual_mask_alignment_mode == "shared_mask_video_flow"
            ),
        ),
        visual_flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        enable_gradient_checkpointing=True,
        skip_action_expert=False,
        visual_loss_weight=float(_env("GWP_VISUAL_LOSS_WEIGHT", "1.0")),
        action_loss_weight=float(_env("GWP_ACTION_LOSS_WEIGHT", "1.0")),
        consequence_loss_weight=consequence_weight,
        consequence_gradient_mode=(
            "policy_only" if consequence_mode == "off" else consequence_mode
        ),
        consequence_interval=int(_env("GWP_CONSEQUENCE_INTERVAL", "4")),
        consequence_start_step=int(_env("GWP_CONSEQUENCE_START_STEP", "0")),
        consequence_visual_sigma=float(_env("GWP_CONSEQUENCE_VISUAL_SIGMA", "1.0")),
        counterfactual_mask_loss_weight=counterfactual_mask_weight,
        counterfactual_mask_compute_control=counterfactual_mask_compute_control,
        counterfactual_mask_bce_weight=float(
            _env("GWP_COUNTERFACTUAL_MASK_BCE_WEIGHT", "1.0")
        ),
        counterfactual_mask_dice_weight=float(
            _env("GWP_COUNTERFACTUAL_MASK_DICE_WEIGHT", "1.0")
        ),
        counterfactual_mask_interval=int(
            _env("GWP_COUNTERFACTUAL_MASK_INTERVAL", "1")
        ),
        counterfactual_mask_start_step=int(
            _env("GWP_COUNTERFACTUAL_MASK_START_STEP", "0")
        ),
        counterfactual_mask_trainable_scope=counterfactual_mask_trainable_scope,
        counterfactual_mask_alignment_mode=counterfactual_mask_alignment_mode,
        counterfactual_mask_probe_pretrained=counterfactual_mask_probe_pretrained,
        counterfactual_action_rank_loss_weight=counterfactual_action_rank_weight,
        counterfactual_action_rank_compute_control=(
            counterfactual_action_rank_compute_control
        ),
        counterfactual_action_rank_margin=counterfactual_action_rank_margin,
        counterfactual_action_rank_min_mse=counterfactual_action_rank_min_mse,
    ),
    optimizers=dict(type="CAME8Bit", lr=6e-5, weight_decay=1e-2),
    schedulers=dict(type="ConstantScheduler"),
    train=dict(
        resume=_bool("GWP_RESUME", False),
        max_epochs=0,
        max_steps=int(_env("GWP_MAX_STEPS", "6000")),
        gradient_accumulation_steps=int(_env("GWP_GRAD_ACCUM_STEPS", "1")),
        mixed_precision="bf16",
        checkpoint_interval=int(_env("GWP_CHECKPOINT_INTERVAL", "1000")),
        checkpoint_total_limit=int(_env("GWP_CHECKPOINT_TOTAL_LIMIT", "-1")),
        checkpoint_save_optimizer=_bool("GWP_CHECKPOINT_SAVE_OPTIMIZER", True),
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        disable_checkpoint=_bool("GWP_DISABLE_CHECKPOINT", False),
        checkpoint_model_dtype=checkpoint_model_dtype,
        strip_optimizer_after_checkpoint=_bool(
            "GWP_STRIP_OPTIMIZER_AFTER_CHECKPOINT", False
        ),
        log_with=None,
        log_interval=int(_env("GWP_LOG_INTERVAL", "20")),
        seed=int(_env("GWP_SEED", "6666")),
        with_ema=with_ema,
        activation_checkpointing=True,
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
