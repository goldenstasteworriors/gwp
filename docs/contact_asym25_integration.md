# Asymmetric25 contact integration gate

The production robot finetune config is
`configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py`.
Importing it runs the executable preflight before the base config or dataloader
is constructed. The checked-in training-gate and robot-sidecar templates are
deliberately incomplete/all-false and cannot authorize training.

## Fixed model route

The config rejects environment attempts to change Q=12, target D=22,
`contact_segment_v1_1`, `hard_st`, B=16, strict semantic routing, strict
sidecars, or required targets. The action route is:

```text
state + reference + text -> contact planner -> canonical hard 22D
    -> 22-to-1024 encoder -> action
```

Action tokens cannot directly attend reference tokens or raw planner hidden.
Action-only inference caches the transported hard semantic prefix. Inference
still returns canonical25D only: this repository has no OmniPicker driver,
reverse calibration mapper, or physical executor, so deployment is No-Go.

## Artifact and review gates

The preflight requires explicit paths for the asymmetric25 dataset, norm file,
hardware manifest, calibration capture, reviewed sidecar root, adjudicated
coverage/segment CSVs, complete inventory, production group split, build
config, and training-gate manifest. It then verifies:

- exact canonical 25D dataset fields and embedded hardware/capture digests;
- v3.1 parquet or legacy JSONL episode IDs and per-episode frame counts;
- non-empty stable dataset ID shared by dataset and contact annotations;
- dataset-native frame clock and equal dataset/sidecar/hardware sample rates;
- norm hardware/capture, dataset-info, episode-metadata and sample-count hashes;
- reviewed/adjudicated/human source CSV hashes, full in-memory re-encoding,
  per-NPZ/provenance hashes, and loader-time NPZ hash rechecks;
- group completeness, no group leakage, and non-empty train/val/test for all
  seeds 6666/7777/8888;
- artifact-bound annotation-agreement, hardware-verification, and matched-hash
  collector reports, plus all separated server/5B gate fields.

The robot sidecar build config starts from
`configs/contact_asym25_robot_sidecar.template.json`. `dataset_id` and
`target_fps` must be filled from the converted robot dataset; the rate must
also equal the hardware manifest's nominal rate. Q=12 and H=48 are fixed.
`group_fields_by_seed` may contain only real subject/session/operator/object/
liquid/batch identifiers. The template's strategy flags remain false until an
independently reviewed production split exists.

## Two-stage boundary

Track A currently describes Ego data at `processed_action_20hz`; it must never
be attached to a robot-native-rate asymmetric25 dataset. The existing
`giga_world_policy_0_5_ego_contact_pretrain.py` is a legacy/smoke config with
environment-selectable v0/v1 schema and no reviewed-production preflight.
Current Track A files contain no formal adjudicated ground truth and incomplete
group metadata, so Stage 1 is also No-Go for scientific training.

A future reviewed Ego pretraining stage and robot-native finetuning stage have
different physical durations for a 48-step horizon. Although start/end tau are
normalized, rate-dependent transfer remains an explicit ablation risk.
