# Asymmetric SONIC G1 action schema

`legacy30` remains the default everywhere. It represents two symmetric
six-channel Inspire hands and remains compatible with the existing 88 episodes.

The opt-in schema ID is:

```text
sonic_g1_rh56dftp6_omnipicker1_v1
```

Its physical state/action width is 25 before zero-padding to 32:

```text
left wrist 9 | right wrist 9 | left RH56DFTP 6 | right OmniPicker aperture 1
```

Wrist actions are local relative transforms. The six RH56DFTP channels and the
OmniPicker aperture are future actuator targets. The two OmniPicker jaws remain
two physical contact surfaces but have one active actuation channel.

## Tensor and time contract

The converted LeRobot table stores one physical state and one physical command
per frame:

```text
observation.state: float32 [25]
action:            float32 [25]
```

The dataset constructs a future action chunk of length `T` (48 in the supplied
config). The transform exposes the following model-facing tensors:

```text
state:              float32 [1, 32]   # dimensions 25:32 are zero
action:             float32 [T, 32]   # dimensions 25:32 are zero
action_dim_mask:    bool    [32]      # first 25 true
action_time_mask:   bool    [T]       # false only for synthesized padded time steps
action_loss_mask:   bool    [T]       # compatibility alias of action_time_mask
action_loss_weight: float32 [T, 32]   # time mask x dimension mask x source weight
```

If the dataset provides `action_time_mask`/`action_loss_mask`, the transform
preserves it. Otherwise it inverts `action_is_pad` (or `<action_key>_is_pad`),
falling back to all-valid only when no padding signal exists. When the transform
itself pads a short chunk, the appended time steps receive zero mask and zero
weight.

For the 20-D human ego adapter only, grip-proxy dimensions 18:20 receive weight
0.25 by default. This does not downweight the RH56DFTP or OmniPicker dimensions.
The current trainer consumes `action_loss_mask`; `action_loss_weight` is an
explicit forward-compatible contract and must be consumed by the trainer before
the 0.25 value can affect optimization.

## Conversion manifest

The asymmetric converter requires source metadata with explicit feature names
and a JSON hardware manifest. Example:

```json
{
  "action_schema_id": "sonic_g1_rh56dftp6_omnipicker1_v1",
  "left_rh56_state_names": [
    "left_little", "left_ring", "left_middle", "left_index",
    "left_thumb_bend", "left_thumb_rotate"
  ],
  "left_rh56_action_names": [
    "left_little", "left_ring", "left_middle", "left_index",
    "left_thumb_bend", "left_thumb_rotate"
  ],
  "right_omnipicker_state_name": "right_omnipicker_aperture",
  "right_omnipicker_action_name": "right_omnipicker_aperture_target"
}
```

The names must exist in `meta/info.json`. Legacy right finger names are rejected
instead of being compressed or truncated into one aperture value.

Convert with:

```bash
python scripts/convert_sonic_wrist_delta.py \
  --input /path/to/new_hardware_lerobot \
  --output /path/to/asymmetric25_lerobot \
  --sonic-root /path/to/GR00T-WholeBodyControl \
  --action-schema asymmetric25 \
  --hardware-manifest /path/to/rh56_omnipicker_manifest.json
```

Compute schema-labelled normalization statistics with no further delta
conversion, because the converter already emits local wrist transforms:

```bash
python scripts/compute_wam_task_norm.py \
  --data-root /path/to/asymmetric25_lerobot \
  --output /path/to/asymmetric25_norm_stats.json \
  --model-dim 32 \
  --delta-mask none \
  --action-schema-id sonic_g1_rh56dftp6_omnipicker1_v1
```

Use `configs/giga_world_policy_0_5_sonic_g1_asymmetric25_finetune.py` for
training. It rejects non-25-D tensors and normalization files without the exact
schema ID.

For inference, add:

```bash
--action-schema asymmetric25
```

The default remains `legacy30`.
