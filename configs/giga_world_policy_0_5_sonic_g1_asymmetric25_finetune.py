"""Opt-in GWP-0.5 finetune for RH56DFTP-left + OmniPicker-right.

Set the regular ``GWP_SONIC_DATA_PATH`` and ``GWP_NORM_STATS_PATH`` variables
to data produced with ``--action-schema asymmetric25``.  Exact source width and
normalization schema checks intentionally reject the legacy 30-D dataset.
"""

import copy
import json
import os
from pathlib import Path

from configs.giga_world_policy_0_5_sonic_g1_finetune import config as base_config
from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
)
from world_action_model.asymmetric_hardware_manifest import (
    calibration_capture_sha256,
    hardware_contract_sha256,
    load_hardware_manifest,
    validate_converted_dataset_info,
    validate_manifest_capture_binding,
)


config = copy.deepcopy(base_config)
config["project_dir"] = os.environ.get(
    "GWP05_PROJECT_DIR",
    "outputs/gwp0_5_sonic_g1_asymmetric25_finetune",
)

hardware_manifest_path = os.environ.get("GWP_ASYMMETRIC_HARDWARE_MANIFEST")
if not hardware_manifest_path:
    raise ValueError(
        "GWP_ASYMMETRIC_HARDWARE_MANIFEST is required for asymmetric25 training"
    )
hardware_manifest = load_hardware_manifest(hardware_manifest_path, mode="training")
calibration_capture_path = os.environ.get("GWP_ASYMMETRIC_CALIBRATION_CAPTURE")
if not calibration_capture_path:
    raise ValueError(
        "GWP_ASYMMETRIC_CALIBRATION_CAPTURE is required for asymmetric25 training"
    )
calibration_capture = json.loads(
    Path(calibration_capture_path).resolve().read_text(encoding="utf-8")
)
validate_manifest_capture_binding(hardware_manifest, calibration_capture)
contract_digest = hardware_contract_sha256(hardware_manifest)
capture_digest = calibration_capture_sha256(calibration_capture)

for dataset_config in config["dataloaders"]["train"]["data_or_config"]:
    dataset_config["robotype"] = "sonic_g1_asymmetric25"
    data_root = Path(dataset_config["data_path"]).resolve()
    info = json.loads((data_root / "meta" / "info.json").read_text(encoding="utf-8"))
    dataset_manifest = info.get("asymmetric_hardware_manifest")
    if not isinstance(dataset_manifest, dict):
        raise ValueError(f"{data_root} is missing asymmetric_hardware_manifest")
    validate_manifest_capture_binding(dataset_manifest, calibration_capture)
    validate_converted_dataset_info(info, dataset_manifest)
    if hardware_contract_sha256(dataset_manifest) != contract_digest:
        raise ValueError(f"{data_root} was converted with a different hardware manifest")
    if info.get("calibration_capture_sha256") != capture_digest:
        raise ValueError(f"{data_root} was converted with a different calibration capture")

transform = config["dataloaders"]["train"]["transform"]
transform.update(
    robotype_to_embodiment_id={"sonic_g1_asymmetric25": 0},
    robotype_default_embodiment_id=0,
    delta_mask_by_embodiment_id={"0": [False] * ASYMMETRIC_ACTION_DIM},
    require_exact_effective_dim=True,
    expected_action_schema_id=ASYMMETRIC_SCHEMA_ID,
    expected_hardware_contract_sha256=contract_digest,
    expected_calibration_capture_sha256=capture_digest,
)
