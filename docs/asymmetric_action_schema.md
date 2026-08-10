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
OmniPicker aperture are future absolute targets in canonical open fraction
(`0=closed, 1=open`). This canonical unit is distinct from device raw units:
the RH56 manual specifies six `POS_SET/POS_ACT` values in `0..2000`, while the
current SONIC Inspire DDS adapter uses normalized `0..1`. A manifest must state
separate measured-state and commanded-target affine mappings, so these cannot
be silently mixed. The two OmniPicker jaws remain two physical contact surfaces
but have one active actuation channel.

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

## Hardware contract and calibration

The asymmetric converter requires both:

- a hardware manifest conforming to
  `schemas/asymmetric25_hardware_manifest.schema.json`;
- a calibration record conforming to
  `schemas/asymmetric25_calibration_capture.schema.json`.

Start from the deliberately invalid templates under `configs/hardware/`. They
contain no fabricated serials, endpoints, ranges, frames or safety values. The
manifest must enumerate the exact six RH56 channels, OmniPicker aperture,
ordered 29 measured-state WBC joints, ordered 29 command WBC joints, both wrist frame conventions, hardware/synchronized
timestamps, latency and safety limits. Its measured-state and command mappings
are separate.

Each calibration measurement contains a structured `result`, not merely a
human `verified` flag. The validator requires the captured `hardware`,
`wrists`, `wbc`, `timing` and `safety` results to match the manifest exactly and
binds their digest into converted data, normalization statistics and inference.

Readiness is fail-closed:

- `documentation_only`: schema/reference work only; cannot convert trainable data.
- `bench_verified`: may convert/train after explicit training approval.
- `robot_verified`: required for deployment, with explicit deployment approval.

The names must exist in `meta/info.json`. Legacy right finger names are rejected
instead of being compressed or truncated into one aperture value.

Convert with:

```bash
python scripts/convert_sonic_wrist_delta.py \
  --input /path/to/new_hardware_lerobot \
  --output /path/to/asymmetric25_lerobot \
  --sonic-root /path/to/GR00T-WholeBodyControl \
  --action-schema asymmetric25 \
  --dataset-id robot_rh56_omnipicker_collection_v1 \
  --hardware-manifest /path/to/rh56_omnipicker_manifest.json \
  --calibration-capture /path/to/rh56_omnipicker_calibration.json
```

`--dataset-id` is mandatory for asymmetric25 and must also appear in the
reviewed robot contact sidecar build config. It prevents a self-consistent
sidecar from another dataset being attached by matching episode numbers alone.

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
training and set both `GWP_ASYMMETRIC_HARDWARE_MANIFEST` and
`GWP_ASYMMETRIC_CALIBRATION_CAPTURE`. The config reads and cross-validates both
files, the converted dataset's embedded contract, and normalization metadata;
it rejects any schema, hardware digest or calibration digest mismatch.

For inference, add:

```bash
--action-schema asymmetric25 \
--hardware-manifest /path/to/robot_verified_manifest.json \
--calibration-capture /path/to/matching_robot_verified_capture.json
```

The default remains `legacy30`.

Inference returns canonical 25D values only. This repository does not contain
the target G1 OmniPicker driver or a calibrated canonical-to-device command
mapper, so even a `robot_verified` manifest cannot by itself actuate the robot.
Until that driver, reverse affine mapping, endpoint integration, watchdog and
bench/robot tests exist, physical robot deployment remains **No-Go**.

## Ego grip polarity

EgoHumanoid source code defines `hand_status` as `0=open, 1=closed`. The legacy
20-D Ego adapter retains that closure proxy unchanged. It is therefore the
inverse of the asymmetric canonical open fraction and is not, by itself, an
OmniPicker command. Any cross-embodiment config that combines Ego input with
the asymmetric schema must set:

```python
ego_grip_semantics="closure_fraction_0_open_1_closed"
```

The transform then performs the explicit `1-status` conversion. An
asymmetric cross-embodiment config without this declaration fails closed;
ordinary legacy Ego configs remain unchanged.

## Current data No-Go

`outputs/grab_red_bottle` contains 88 legacy dual-Inspire episodes, not the
target RH56DFTP/OmniPicker hardware. Its right side has six named finger
channels (`little`, `ring`, `middle`, `index`, `thumb_bend`, `thumb_rotate`),
so it is audit-only and is rejected by the asymmetric converter. See
`docs/audits/grab_red_bottle_asymmetric25_audit.md`.

## Representation boundary from prior work

The local Zotero/Obsidian notes for OmniUMI, CrossDex, EgoScale and motion
retargeting support a shared intermediate representation only when the
embodiment-specific feasibility/retargeting map remains explicit. They do not
support renaming six legacy right-hand joints as a one-DoF gripper. Relevant
primary sources are [CrossDex](https://arxiv.org/abs/2410.02479),
[TopoRetarget](https://arxiv.org/abs/2606.16272),
[Transferring Contact, Not Just Motion](https://arxiv.org/abs/2606.15516) and
[DexUMI](https://github.com/real-stanford/DexUMI). This contract is therefore an
action adapter/safety boundary for a contact-transition representation, not an
independent universal canonical latent.
