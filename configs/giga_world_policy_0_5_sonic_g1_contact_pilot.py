"""Strict contact-transition bottleneck pilot for SONIC/G1 fine-tuning.

This config inherits the existing fine-tune recipe.  It can run without
contact labels as a latent-bottleneck smoke/pilot; set
``GWP_CONTACT_REQUIRE_TARGETS=1`` for supervised experiments.
"""

import copy
import os

from configs.giga_world_policy_0_5_sonic_g1_finetune import config as base_config


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value


config = copy.deepcopy(base_config)
config["project_dir"] = _env(
    "GWP05_PROJECT_DIR",
    "outputs/gwp0_5_sonic_g1_contact_pilot",
)

num_contact_tokens = int(_env("GWP_CONTACT_TOKENS", "12"))
contact_target_dim = int(_env("GWP_CONTACT_TARGET_DIM", "50"))

config["models"]["transformer"].update(
    num_contact_tokens=num_contact_tokens,
    contact_target_dim=contact_target_dim,
    contact_condition_dim=contact_target_dim,
    contact_bottleneck=_env("GWP_CONTACT_STRICT_BOTTLENECK", "1") == "1",
)
config["models"].update(
    contact_loss_weight=float(_env("GWP_CONTACT_LOSS_WEIGHT", "0.25")),
    contact_loss_type=_env("GWP_CONTACT_LOSS_TYPE", "bce"),
    contact_condition_mode=_env("GWP_CONTACT_CONDITION_MODE", "predicted"),
    require_contact_targets=_env("GWP_CONTACT_REQUIRE_TARGETS", "0") == "1",
)
