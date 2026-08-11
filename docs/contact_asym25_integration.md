# Asymmetric25 contact integration gate

The production robot finetune config is
`configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py`.
Importing it runs the executable preflight before the base config or dataloader
is constructed. The checked-in training-gate and robot-sidecar templates are
deliberately incomplete/all-false and cannot authorize training.

## Fixed model route

The config rejects environment attempts to change Q=12, target D=22,
`contact_segment_v1_1`, B=16, strict semantic routing, strict sidecars, or
required targets. It requires an explicit `GWP_MATCHED_CONTROL_VARIANT_ID`
and derives the only allowed transport override from it:
`soft_semantic -> soft`, `hard_st -> hard_st`, or
`hard_detached -> hard_detached`. The action route is:

```text
state + reference + text -> contact planner -> selected canonical 22D transport
    -> 22-to-1024 encoder -> action
```

Action tokens cannot directly attend reference tokens or raw planner hidden.
Action-only inference caches the transported semantic prefix. Inference
still returns canonical25D only: this repository has no OmniPicker driver,
reverse calibration mapper, or physical executor, so deployment is No-Go.

## Metadata-only hardware discovery

Before any device-side probe, run the host discovery helper on the robot computer:

```bash
python -B scripts/discover_asymmetric25_hardware.py
```

The helper only accepts repository-contained regular non-symlink JSON files and
the allowlisted `/sys/class/net` and `/dev/serial/by-id` metadata roots. It pins
each existing metadata root with a no-follow directory file descriptor and
performs enumeration, attribute reads, and raw link reads relative to pinned
directory descriptors. Serial link targets are never resolved or opened. Root
replacement and symlink substitution are rejected. Within these validated
inputs it does not open a serial, CAN, DDS, Modbus, or network endpoint and
cannot send a control command. Arbitrary filesystem roots and special/FIFO/
socket/symlink JSON inputs are rejected before content reads.

Default JSON output contains only interface/link counts and coarse categories;
it emits neither raw local identifiers nor stable identifier digests that could
be dictionary-recovered or linked across reports. It also redacts the kernel,
absolute paths and execution time, and reports both generic empty-value
placeholders and bounded schema-structure/finite-number/bounds issues. JSON
templates reject `NaN`, positive/negative infinity, and overflow-to-infinity
numbers. This audit is not a replacement for
`validate_asymmetric_hardware_manifest.py`. Exact local identifiers and raw
serial-link text may be shown only with `--include-sensitive-local-metadata`;
that output is local-only, may contain device serials and host fingerprints, and
must never be committed or attached to a public report.

Empty candidate lists are only a statement about names visible to this host at
that instant; they are not evidence that hardware is absent. The report is not a
calibration capture and never authorizes training or deployment. An operator must
still identify mounted devices from physical labels and official vendor
documentation, then measure and review all values.

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

The robot gate is `contact-asym25-training-gate/v2`; v1 is rejected. Its JSON,
all evidence reports, and the six A/B/adjudicated CSV inputs are opened once
with no-follow regular-file verification. Duplicate JSON keys, `NaN`,
infinities, overflow such as `1e999`, boolean numeric fields, expired evidence,
and a runtime nonce not found at the selected variant in the gate's reviewed
identity mapping fail closed. The stored agreement summary is
not trusted: preflight re-runs the existing formal v1.1 A/B reducer from the
verified CSV bytes and checks the adjudicated parent/third-reviewer contract.

The gate additionally binds a `matched-control-hash-collector/v2` report,
trace-file manifest, global/hash-schema receipts, current HEAD collector bytes,
launcher topology, and an external reviewer acceptance. Each transport variant
uses a distinct run ID, run nonce, and external acceptance. The comparator
requires a single shared model hash after replacing exactly the transport-mode
field; any other model/config difference fails. For one production split seed,
the offline comparison command is:

```bash
conda run -n gigaworld-policy python -B scripts/compare_matched_control_hashes.py \
  --variant soft_semantic=/ABS/PATH/soft_traces \
  --variant hard_st=/ABS/PATH/hard_st_traces \
  --variant hard_detached=/ABS/PATH/hard_detached_traces \
  --required-seed 6666 \
  --distinct-variant-run-identities \
  --output-report /ABS/PROJECT/EVIDENCE/matched_control_seed6666.json
```

换行位置：每个行尾反斜杠后换行；把 seed 和所有 `/ABS/...` 路径替换为本次运行的真实值，输出必须位于项目盘而不是 `/tmp`。
报告路径采用独占创建；若已存在会直接拒绝，不能覆盖审计证据。

Production config import no longer accepts a detached old report SHA as a
substitute for current capture. It requires `GWP_MATCHED_CONTROL_OUTPUT_DIR`
and binds the current trace rows to the current preflight, training-code and
dataset-tree receipts, initial/base weights, bf16/action/contact schema,
optimizer/scheduler/training hashes, runtime package/topology receipt, and
train-window contract. The structured training-config receipt retains the
actual seed and its full hash; a second stable hash replaces only the seed with
the literal `__CONTACT_ASYM25_SPLIT_SEED__`, so the nine-run aggregate can
prove that different seeds introduce no hidden training-config changes. The
required current-run identities are:

```bash
export GWP_CONTACT_TRAINING_RUN_ID="RUN_ID_AT_LEAST_16_CHARS"
export GWP_CONTACT_TRAINING_RUN_NONCE="RUN_NONCE_AT_LEAST_16_CHARS"
export GWP_MATCHED_CONTROL_RUN_ID="$GWP_CONTACT_TRAINING_RUN_ID"
export GWP_MATCHED_CONTROL_RUN_NONCE="$GWP_CONTACT_TRAINING_RUN_NONCE"
export GWP_MATCHED_CONTROL_VARIANT_ID="hard_st"
export GWP_MATCHED_CONTROL_SEED="6666"
export GWP_MATCHED_CONTROL_DATASET_ID="REVIEWED_DATASET_ID"
export GWP_MATCHED_CONTROL_OUTPUT_DIR="/ABS/PROJECT/OUTPUT/current_run_traces"
```

换行位置：以上每条 `export` 独占一行；`RUN_*`、transport variant、seed、dataset ID 和绝对输出路径必须替换为 gate/launcher 的真实绑定值。三个 transport 分别导入同一个共享 preflight receipt；variant、run ID/nonce 和外部 acceptance 则分别取 gate expectations 的对应映射值。

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

The fail-closed Ego entry and its separate 20 Hz fingerprint/norm contract are
documented in `docs/ego_discrete_contact_stage1.md`. It deliberately requires
no robot hardware manifest; attaching an Ego sidecar to the robot-native stage
or a robot sidecar to this Ego stage fails clock/dataset identity checks.
