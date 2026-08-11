"""Route-aware runtime adapter for causal-main held-out evaluation.

The mature asymmetric25 runtime owns dataset transforms, VAE setup and action
denoising.  This subclass only adapts the three causal output layouts while
keeping the same fixed-window inference path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from world_action_model.asymmetric_action_schema import ASYMMETRIC_ACTION_DIM
from world_action_model.contact_asym25_eval_runtime import ProductionHeldoutRuntime
from world_action_model.contact_asym25_heldout_eval import (
    HORIZON_FRAMES,
    MODEL_ACTION_DIM,
    HeldoutEvaluationError,
)
from world_action_model.contact_causal_production import (
    ensure_exact_causal_route,
    validate_checkpoint_transformer_semantics,
)
from world_action_model.contact_causal_heldout_eval import (
    INFERENCE_PACKET_KEYS,
    validate_target_free_inference_packet,
)


METRIC_TARGET_PACKET_KEYS = frozenset(
    {
        "window_id",
        "action",
        "action_dim_mask",
        "action_time_mask",
        "action_denorm_scale",
        "contact_target",
        "contact_loss_mask",
        "contact_confidence",
        "contact_condition_mask",
    }
)


class CausalMetricTargetReader:
    """Verified target-only reader used after the prediction process exits."""

    def __init__(
        self,
        *,
        preflight_receipt: Mapping[str, Any],
        verified_private_cache_parent: str,
        verified_run_nonce: str,
    ) -> None:
        import torch

        from world_action_model.asymmetric_action_schema import (
            ASYMMETRIC_SCHEMA_ID,
        )
        from world_action_model.contact_asym25_heldout_eval import MODEL_ACTION_DIM
        from world_action_model.contact_causal_preflight import (
            build_robot_runtime_source_receipts_v1,
        )
        from world_action_model.datasets.wam_lerobot_dataset import WAMLeRobotDataset
        from world_action_model.transforms.wa_transforms_lerobot_pretrain import (
            WALeRobotTransformsPretrain,
        )

        self.torch = torch
        self.preflight = dict(preflight_receipt)
        data_root = Path(self.preflight["data_root"])
        runtime_sources = build_robot_runtime_source_receipts_v1(
            hardened_preflight_receipt=self.preflight,
            t5_embedding_dir=data_root / "t5_embedding",
        )
        self.dataset = WAMLeRobotDataset(
            data_path=str(data_root),
            data_size=None,
            delta_info={"action": HORIZON_FRAMES},
            force_cache_sync=False,
            training_source_runtime_receipt=runtime_sources[
                "training_source_runtime_receipt"
            ],
            t5_runtime_source_receipt=runtime_sources["t5_runtime_source_receipt"],
            verified_private_cache_parent=verified_private_cache_parent,
            verified_training_run_nonce=verified_run_nonce,
            source_column_projection=(
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "task_index",
                "observation.state",
                "action",
            ),
            source_column_projection_allow_targets=True,
            skip_video_decoding=True,
            video_backend="pyav",
            t5_load_from="dir",
            t5_cfg={
                "t5_embedding_dir": str(data_root / "t5_embedding"),
                "t5_embedding_pattern": "episode_{episode_index:06d}.pt",
                "t5_embedding_key": "t5_embedding",
            },
            t5_cache_size=128,
            contact_sidecar_dir=self.preflight["sidecar_dir"],
            contact_sidecar_pattern="episode_{episode_index:06d}.npz",
            contact_sidecar_cache_size=8,
            contact_sidecar_strict=True,
            contact_sidecar_require_reviewed_v1_1=True,
            contact_sidecar_expected_meta_sha256=self.preflight[
                "sidecar_meta_sha256"
            ],
            contact_sidecar_expected_episode_sha256=self.preflight[
                "episode_npz_sha256"
            ],
            robotype="sonic_g1_asymmetric25",
        )
        norm_path = Path(self.preflight["norm_stats_path"])
        self.transform = WALeRobotTransformsPretrain(
            is_train=False,
            eval_crop_mode="center",
            dst_size=(320, 384),
            num_frames=HORIZON_FRAMES,
            fps=float(self.preflight["dataset_fps"]),
            norm_path=[{
                "path": str(norm_path),
                "data_paths": [str(data_root)],
                "sha256": self.preflight["norm_stats_sha256"],
            }],
            robotype_to_embodiment_id={"sonic_g1_asymmetric25": 0},
            robotype_default_embodiment_id=0,
            model_action_dim=MODEL_ACTION_DIM,
            delta_mask_by_embodiment_id={"0": [False] * ASYMMETRIC_ACTION_DIM},
            require_exact_effective_dim=True,
            expected_action_schema_id=ASYMMETRIC_SCHEMA_ID,
            expected_hardware_contract_sha256=self.preflight[
                "hardware_contract_sha256"
            ],
            expected_calibration_capture_sha256=self.preflight[
                "calibration_capture_sha256"
            ],
            norm_use_quantiles=True,
            norm_enable_clamp=True,
            num_views=1,
            view_keys=["observation.images.ego_view"],
            image_cfg={"mask_generator": {"max_ref_frames": 1, "start": 1, "factor": 4}},
            image_augmentation={"enabled": False},
            max_prompt_len=64,
            subtask_prob=0,
            matched_control_capture=False,
            emit_action_denorm_scale=True,
            emit_contact_condition_inputs=True,
        )
        self.dataset.open()
        columns = tuple(self.dataset.dataset.hf_dataset.column_names)
        expected_columns = tuple(self.dataset.dataset._loaded_source_columns or ())
        if columns != expected_columns or "action" not in columns:
            raise HeldoutEvaluationError(
                "metric target Arrow schema was not source-projected before loading"
            )
        self.source_column_receipt = {
            "reader_role": "post_prediction_metric_targets_only",
            "arrow_schema_columns": list(columns),
            "action_column_read": True,
            "contact_sidecar_read": True,
            "model_loaded": False,
        }

    @staticmethod
    def _scalar_int(value: object, name: str) -> int:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise HeldoutEvaluationError(f"metric target {name} is invalid")
        return int(value)

    def load(self, window: Mapping[str, Any]) -> dict[str, Any]:
        raw = self.dataset._get_data(int(window["global_index"]))
        if (
            self._scalar_int(raw.get("episode_index"), "episode")
            != int(window["episode_id"])
            or self._scalar_int(raw.get("frame_index"), "frame")
            != int(window["frame_index"])
            or self._scalar_int(raw.get("index"), "index")
            != int(window["global_index"])
        ):
            raise HeldoutEvaluationError("metric target fixed-window identity differs")
        raw = dict(raw)
        raw["observation.images.ego_view"] = self.torch.zeros(
            (5, 3, 384, 320), dtype=self.torch.uint8
        )
        transformed = self.transform(raw)
        packet = {
            "window_id": window["window_id"],
            **{
                key: transformed[key]
                for key in METRIC_TARGET_PACKET_KEYS - {"window_id"}
            },
        }
        validate_metric_target_packet(packet)
        return packet

    def close(self) -> None:
        self.dataset.close()


def validate_metric_target_packet(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != METRIC_TARGET_PACKET_KEYS:
        raise HeldoutEvaluationError("causal metric-target packet schema differs")
    if not isinstance(value["window_id"], str) or not value["window_id"]:
        raise HeldoutEvaluationError("causal metric-target window_id is invalid")
    return value


class CausalProductionHeldoutRuntime(ProductionHeldoutRuntime):
    """Run the mature action evaluator and expose the exact route packet."""

    def __init__(
        self,
        *,
        verified_private_cache_parent: str,
        verified_run_nonce: str,
        **kwargs: Any,
    ) -> None:
        from world_action_model.contact_causal_preflight import (
            build_robot_runtime_source_receipts_v1,
        )

        hardened = kwargs["preflight_receipt"]
        data_root = Path(hardened["data_root"])
        runtime_sources = build_robot_runtime_source_receipts_v1(
            hardened_preflight_receipt=hardened,
            t5_embedding_dir=data_root / "t5_embedding",
        )
        super().__init__(
            open_metric_dataset=False,
            force_cache_sync=False,
            training_source_runtime_receipt=runtime_sources[
                "training_source_runtime_receipt"
            ],
            t5_runtime_source_receipt=runtime_sources[
                "t5_runtime_source_receipt"
            ],
            verified_private_cache_parent=verified_private_cache_parent,
            verified_training_run_nonce=verified_run_nonce,
            **kwargs,
        )
        from world_action_model.datasets.wam_lerobot_dataset import WAMLeRobotDataset

        # The mature dataset remains unopened at the sample level until metrics
        # are scored.  This second view requests image deltas only: it never
        # queries a future action chunk and has no contact-sidecar configuration.
        data_root = Path(self.preflight["data_root"])
        self.inference_dataset = None
        try:
            self.inference_dataset = WAMLeRobotDataset(
                data_path=str(data_root),
                data_size=None,
                delta_info=None,
                delta_frames={
                    "observation.images.ego_view": [0, 12, 24, 36, 48]
                },
                force_cache_sync=False,
                training_source_runtime_receipt=runtime_sources[
                    "training_source_runtime_receipt"
                ],
                t5_runtime_source_receipt=runtime_sources[
                    "t5_runtime_source_receipt"
                ],
                verified_private_cache_parent=verified_private_cache_parent,
                verified_training_run_nonce=verified_run_nonce,
                video_backend="pyav",
                t5_load_from="dir",
                t5_cfg={
                    "t5_embedding_dir": str(data_root / "t5_embedding"),
                    "t5_embedding_pattern": "episode_{episode_index:06d}.pt",
                    "t5_embedding_key": "t5_embedding",
                },
                t5_cache_size=128,
                robotype="sonic_g1_asymmetric25",
                source_column_projection=(
                    "index",
                    "episode_index",
                    "frame_index",
                    "timestamp",
                    "task_index",
                    "observation.state",
                    "observation.images.ego_view",
                ),
            )
            self.inference_dataset.open_target_free_inference_view()
            self.prediction_source_column_receipt = {
                "reader_role": "prediction_inputs_only",
                "arrow_schema_columns": list(
                    self.inference_dataset._target_free_source_columns
                ),
                "removed_target_columns": list(
                    self.inference_dataset._target_free_removed_source_columns
                ),
                "action_column_read": False,
                "contact_sidecar_read": False,
            }
        except BaseException:
            if self.inference_dataset is not None:
                self.inference_dataset.close()
                self.inference_dataset = None
            super().close()
            raise
        self._pending_window_id: str | None = None
        self._prediction_completed_for_window = False
        self._pending_safe_inputs: dict[str, Any] | None = None

    def _assert_window_identity(
        self, raw: Mapping[str, Any], window: Mapping[str, Any]
    ) -> None:
        episode = self._scalar_int(raw.get("episode_index"), "episode_index")
        frame = self._scalar_int(raw.get("frame_index"), "frame_index")
        authoritative_index = self._scalar_int(raw.get("index"), "global index")
        if (
            episode != int(window["episode_id"])
            or frame != int(window["frame_index"])
            or authoritative_index != int(window["global_index"])
        ):
            raise HeldoutEvaluationError(
                "fixed window identity differs from the target-free runtime sample"
            )

    def load_inference_window(self, window: Mapping[str, Any]) -> dict[str, Any]:
        """Load and transform model inputs without loading future/contact targets."""

        random = __import__("random")
        random.seed(0)
        np.random.seed(0)
        self.torch.manual_seed(0)
        raw = self.inference_dataset._get_data(int(window["global_index"]))
        self._assert_window_identity(raw, window)
        if any(key.startswith("contact_") for key in raw):
            raise HeldoutEvaluationError(
                "target-free dataset unexpectedly loaded contact-sidecar fields"
            )
        if any(
            key == "action"
            or key.startswith("action_")
            or key.startswith("contact_")
            for key in raw
        ):
            raise HeldoutEvaluationError(
                "target-free projected source emitted an action/contact target"
            )
        # A canonical zero placeholder supplies transform shape metadata; it
        # is generated after the source projection and contains no data bytes.
        model_raw = dict(raw)
        model_raw["action"] = self.torch.zeros(
            (HORIZON_FRAMES, ASYMMETRIC_ACTION_DIM), dtype=self.torch.float32
        )
        model_raw["action_is_pad"] = self.torch.zeros(
            HORIZON_FRAMES, dtype=self.torch.bool
        )
        transformed = self.transform(model_raw)
        packet = {
            "window_id": window["window_id"],
            "ref_images": transformed["ref_images"],
            "prompt_embeds": transformed["prompt_embeds"],
            "state": transformed["state"],
            "embodiment_id": transformed["embodiment_id"],
            "action_dim_mask": transformed["action_dim_mask"],
        }
        validate_target_free_inference_packet(packet)
        self._pending_window_id = window["window_id"]
        self._prediction_completed_for_window = False
        self._pending_safe_inputs = {
            key: packet[key].detach().cpu().clone()
            for key in INFERENCE_PACKET_KEYS - {"window_id"}
        }
        return packet

    def load_metric_targets(self, window: Mapping[str, Any]) -> dict[str, Any]:
        """Prediction workers are structurally unable to load metric targets."""

        del window
        raise HeldoutEvaluationError(
            "metric targets require the independent post-prediction scorer process"
        )

    def load_checkpoint(
        self, checkpoint_path: str, *, checkpoint_receipt: Mapping[str, Any]
    ) -> None:
        from diffusers import FlowMatchEulerDiscreteScheduler
        from world_action_model.contact_causal_checkpoint import (
            load_pinned_safe_weight_mapping,
        )
        from world_action_model.contact_causal_production import (
            model_parameter_and_buffer_digests,
        )
        from world_action_model.contact_transition_schema import (
            SEGMENT_SCHEMA_VERSION,
            SEGMENT_TARGET_DIM,
        )
        from world_action_model.models import CasualWorldActionTransformer_MoT
        from world_action_model.verified_io import read_verified_bytes, strict_json_loads

        self.unload_checkpoint()
        checkpoint_root = Path(checkpoint_path).expanduser().absolute()
        weight_entry = checkpoint_receipt["checkpoint_file"]
        weight_path = Path(weight_entry["path_record_only"]).absolute()
        if weight_path.parent != checkpoint_root:
            raise HeldoutEvaluationError(
                "causal checkpoint receipt path differs from inference path"
            )
        config_payload, config_sha256, _ = read_verified_bytes(
            checkpoint_root / "config.json",
            expected_sha256=checkpoint_receipt["checkpoint_artifact_hashes"][
                "config_sha256"
            ],
        )
        config = strict_json_loads(
            config_payload, name="pinned causal inference checkpoint config"
        )
        if not isinstance(config, dict) or config_sha256 != checkpoint_receipt[
            "checkpoint_artifact_hashes"
        ]["config_sha256"]:
            raise HeldoutEvaluationError("pinned causal checkpoint config differs")
        route = ensure_exact_causal_route(config.get("contact_causal_route"))
        validate_checkpoint_transformer_semantics(config, route=route)
        mapping = load_pinned_safe_weight_mapping(
            weight_path,
            expected_sha256=weight_entry["sha256"],
            expected_serialization=weight_entry["serialization"],
        )
        model = CasualWorldActionTransformer_MoT.from_config(config)
        incompatible = model.load_state_dict(mapping, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise HeldoutEvaluationError(
                "pinned causal checkpoint tensor mapping differs from its model"
            )
        parameter_sha256, buffer_sha256 = model_parameter_and_buffer_digests(model)
        if (
            parameter_sha256
            != checkpoint_receipt["parameter_topology"]["parameter_tensor_sha256"]
            or buffer_sha256
            != checkpoint_receipt["parameter_topology"]["buffer_tensor_sha256"]
        ):
            raise HeldoutEvaluationError(
                "pinned causal inference tensors differ from reviewed topology"
            )
        direct = route == "aux_only_direct_q12"
        expected = {
            "in_action_channels": MODEL_ACTION_DIM,
            "out_action_channels": MODEL_ACTION_DIM,
            "num_contact_tokens": 12,
            "contact_target_dim": SEGMENT_TARGET_DIM,
            "contact_condition_dim": SEGMENT_TARGET_DIM,
            "contact_bottleneck": not direct,
            "contact_schema_version": SEGMENT_SCHEMA_VERSION,
            "contact_semantic_bottleneck": not direct,
            "contact_semantic_transport_mode": "hard_st",
            "contact_time_bins": 16,
            "production_matrix_id": "contact-asym25-causal-main-q12-b16-stage2/v1",
        }
        for name, value in expected.items():
            if getattr(model.config, name, None) != value:
                raise HeldoutEvaluationError(
                    f"causal checkpoint model config {name} differs"
                )
        model.requires_grad_(False)
        model.eval().to(self.device, dtype=self.dtype)
        model._enable_action_only_prefix_cache = True
        self.transformer = model
        self.causal_route = route
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)

    def predict(
        self, transformed: Mapping[str, Any], *, action_noise_seed: int
    ) -> dict[str, np.ndarray | None]:
        validate_target_free_inference_packet(transformed)
        if self._pending_window_id != transformed["window_id"]:
            raise HeldoutEvaluationError(
                "causal inference packet is stale or was not runtime-loaded"
            )
        torch = self.torch
        if self.transformer is None or self.scheduler is None:
            raise HeldoutEvaluationError("no causal checkpoint is loaded")
        ref_latents = self._encode_reference(transformed["ref_images"])
        state = transformed["state"].unsqueeze(0).to(self.device, dtype=self.dtype)
        prompt = transformed["prompt_embeds"].unsqueeze(0).to(
            self.device, dtype=self.dtype
        )
        embodiment_id = transformed["embodiment_id"].reshape(1).to(self.device)
        dim_mask = transformed["action_dim_mask"].to(self.device, dtype=self.dtype)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(action_noise_seed))
        action = torch.randn(
            (1, HORIZON_FRAMES, MODEL_ACTION_DIM),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        action = action * dim_mask[None, None, :]
        patch = tuple(int(value) for value in self.transformer.config.patch_size)
        if any(
            int(ref_latents.shape[axis + 2]) % patch[axis] != 0
            for axis in range(3)
        ):
            raise HeldoutEvaluationError("reference latent shape is not patch divisible")
        num_ref_tokens = int(
            np.prod(
                [
                    int(ref_latents.shape[axis + 2]) // patch[axis]
                    for axis in range(3)
                ]
            )
        )
        num_state_tokens = int(state.shape[1])
        timestep_template = torch.zeros(
            (1, num_state_tokens + num_ref_tokens + HORIZON_FRAMES),
            device=self.device,
            dtype=self.dtype,
        )
        empty_noisy_latents = ref_latents[:, :, :0]
        self.scheduler.set_timesteps(self.inference_steps, device=self.device)
        first_logits = None
        first_aux_hard = None
        first_packet = None
        self.transformer.reset_action_only_prefix_cache()
        try:
            with torch.inference_mode():
                for index, timestep_value in enumerate(self.scheduler.timesteps):
                    timestep = timestep_template.clone()
                    timestep[:, num_state_tokens + num_ref_tokens :] = (
                        timestep_value.to(timestep.dtype)
                    )
                    output = self.transformer(
                        ref_latents=ref_latents,
                        noisy_latents=empty_noisy_latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt,
                        return_dict=True,
                        action=action,
                        state=state,
                        embodiment_id=embodiment_id,
                        action_only=True,
                    )
                    if not isinstance(output, dict) or "action_pred" not in output:
                        raise HeldoutEvaluationError(
                            "causal action-only checkpoint omitted action prediction"
                        )
                    if index == 0:
                        if "contact_pred" not in output:
                            raise HeldoutEvaluationError(
                                "first causal action-only pass omitted planner logits"
                            )
                        first_logits = output["contact_pred"].detach().clone()
                        first_aux_hard = self.transformer.transport_contact_semantic(
                            self.transformer.activate_contact_segment_logits(
                                first_logits
                            )
                        ).detach()
                        if self.causal_route == "equal_bit_unstructured_hard_st":
                            if "contact_code" not in output:
                                raise HeldoutEvaluationError(
                                    "equal-bit route omitted its action-visible code"
                                )
                            first_packet = output["contact_code"].detach().clone()
                        elif self.causal_route == "hard_st_semantic":
                            if "contact_semantic" not in output:
                                raise HeldoutEvaluationError(
                                    "semantic route omitted its action-visible packet"
                                )
                            first_packet = output["contact_semantic"].detach().clone()
                        elif {"contact_code", "contact_semantic"} & set(output):
                            raise HeldoutEvaluationError(
                                "direct route exposed an action-visible contact packet"
                            )
                    else:
                        packet = output.get(
                            "contact_code"
                            if self.causal_route == "equal_bit_unstructured_hard_st"
                            else "contact_semantic"
                        )
                        if first_packet is not None and (
                            packet is None
                            or not torch.equal(first_packet, packet.detach())
                        ):
                            raise HeldoutEvaluationError(
                                "cached causal packet changed during action denoising"
                            )
                        if self.causal_route == "aux_only_direct_q12" and packet is not None:
                            raise HeldoutEvaluationError(
                                "direct route fabricated a cached packet"
                            )
                    action = self.scheduler.step(
                        output["action_pred"], timestep_value, action, return_dict=False
                    )[0]
                    action = action * dim_mask[None, None, :]
        finally:
            self.transformer.reset_action_only_prefix_cache()
        if first_logits is None or first_aux_hard is None:
            raise HeldoutEvaluationError("causal evaluator did not observe planner output")
        result = {
            "action_prediction": action[0].float().cpu().numpy(),
            "contact_logits": first_logits[0].float().cpu().numpy(),
            "contact_hard_semantic": first_aux_hard[0].float().cpu().numpy(),
            "route_packet": (
                None
                if first_packet is None
                else first_packet[0].float().cpu().numpy()
            ),
        }
        self._prediction_completed_for_window = True
        return result

    def close(self) -> None:
        if getattr(self, "inference_dataset", None) is not None:
            self.inference_dataset.close()
            self.inference_dataset = None
        super().close()
