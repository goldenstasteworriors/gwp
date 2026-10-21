"""Default SONIC fine-tuning with measured right joints and optional visual memory."""
import copy
import os
from configs.giga_world_policy_0_5_sonic_g1_finetune import config as base_config

config = copy.deepcopy(base_config)
view = "observation.images.right_wrist"
memory = os.environ.get("GWP_SHORT_MEMORY", "0") == "1"
history = list(range(-120, 0, 15))  # Four seconds at 2 Hz; current frame follows.
dataset = config["dataloaders"]["train"]["data_or_config"][0]
dataset["delta_frames"] = {view: (history if memory else []) + [0, 12, 24, 36, 48]}
transform = config["dataloaders"]["train"]["transform"]
transform.update(view_keys=[view], delta_mask_by_embodiment_id={"0": [False] * 8})
if memory:
    transform.update(type="WALeRobotShortMemoryTransforms", history_frames=len(history))
    config["models"]["short_memory_spec"] = dict(
        style="cortex_mem_inspired_wan_causal_video_latents",
        camera=view, history_frame_offsets=history, fps=30,
        include_current_in_memory_encoder=True,
        separate_current_reference=True,
    )
