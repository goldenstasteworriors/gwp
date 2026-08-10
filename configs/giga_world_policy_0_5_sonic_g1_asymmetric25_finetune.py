"""Opt-in GWP-0.5 finetune for RH56DFTP-left + OmniPicker-right.

Set the regular ``GWP_SONIC_DATA_PATH`` and ``GWP_NORM_STATS_PATH`` variables
to data produced with ``--action-schema asymmetric25``.  Exact source width and
normalization schema checks intentionally reject the legacy 30-D dataset.
"""

import copy
import os

from configs.giga_world_policy_0_5_sonic_g1_finetune import config as base_config
from world_action_model.asymmetric_action_schema import (
    ASYMMETRIC_ACTION_DIM,
    ASYMMETRIC_SCHEMA_ID,
)


config = copy.deepcopy(base_config)
config["project_dir"] = os.environ.get(
    "GWP05_PROJECT_DIR",
    "outputs/gwp0_5_sonic_g1_asymmetric25_finetune",
)

for dataset_config in config["dataloaders"]["train"]["data_or_config"]:
    dataset_config["robotype"] = "sonic_g1_asymmetric25"

transform = config["dataloaders"]["train"]["transform"]
transform.update(
    robotype_to_embodiment_id={"sonic_g1_asymmetric25": 0},
    robotype_default_embodiment_id=0,
    delta_mask_by_embodiment_id={"0": [False] * ASYMMETRIC_ACTION_DIM},
    require_exact_effective_dim=True,
    expected_action_schema_id=ASYMMETRIC_SCHEMA_ID,
)
