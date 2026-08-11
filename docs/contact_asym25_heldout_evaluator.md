# Production contact/asymmetric25 held-out evaluator

The production contract is deliberately single-seed. One persisted preflight
receipt selects exactly one `GWP_SPLIT_SEED` from `6666`, `7777`, or `8888` and
binds its exact train/val/test episode IDs. A training run, its matched-control
variants, its fixed windows, its step 0/100/200 checkpoints, and its held-out
report must all carry that same receipt identity. A receipt is never reused for
another seed.

The legacy configs do not provide the production receipt arguments and retain
their previous unfiltered dataset behavior.

## Per-seed sequence

Set the existing production preflight variables plus `GWP_SPLIT_SEED`, then
persist the replayable receipt before constructing a training dataset:

```bash
conda run -n gigaworld-policy python -B scripts/preflight_contact_asym25_finetune.py \
  --output evidence/contact_asym25/preflight_seed6666.json
```

Create the deterministic, augmentation-free window manifest. It contains
exactly 256 full-horizon validation windows and 256 full-horizon test windows
from the receipt-selected group split. Selection is SHA-256 ranked without
replacement; action padding is forbidden.

```bash
conda run -n gigaworld-policy python -B scripts/evaluate_contact_asym25_heldout.py \
  prepare-windows \
  --preflight-receipt evidence/contact_asym25/preflight_seed6666.json \
  --output evidence/contact_asym25/windows_seed6666.json
```

Production config import additionally requires these two paths. The config
replays both artifacts, passes only the exact train IDs to `FastLeRobotDataset`
before it opens any samples. Production normalization is separately bound to
that seed's train IDs; the old all-66 statistics are audit/legacy evidence only.
The run identity is explicit and cannot be inferred from an output directory.

```bash
export GWP_CONTACT_ASYM25_PREFLIGHT_RECEIPT="$PWD/evidence/contact_asym25/preflight_seed6666.json"
export GWP_CONTACT_HELDOUT_WINDOW_MANIFEST="$PWD/evidence/contact_asym25/windows_seed6666.json"
export GWP_CONTACT_TRAINING_RUN_NONCE="REVIEWED_UNIQUE_RUN_NONCE"
export GWP_CONTACT_TRAINING_RUN_ID="REVIEWED_UNIQUE_RUN_ID"
export GWP_CONTACT_EXPERIMENT_FAMILY_ID="CONTACT_ASYM25_FAMILY_2026"
export GWP_CONTACT_VARIANT_ID="contact_asym25_hard_st"
export GWP_CONTACT_MATCHED_CONTROL_RECEIPT_SHA256="64_HEX_REVIEWED_CONTROL_RECEIPT"
```

Step 0 is saved before optimization. Periodic saving is fixed at 100 steps, so
step 100 and step 200 are present. The evaluator's `checkpoint_path` is the
saved transformer subdirectory (for example
`checkpoint_epoch_1_step_100/transformer`). The trainer writes a non-circular
bundle manifest (model/config bytes only) and a training claim receipt there.
Step 0 binds the initial-weights tree; step 100 binds step 0, and step 200 binds
step 100. The trainer never writes reviewer acceptance.

In a separate reviewer process/output directory, rehash each checkpoint and
create one external acceptance. The identity and nonce are reviewer-authored;
do not reuse a nonce between checkpoints.

```bash
conda run -n gigaworld-policy python -B scripts/review_contact_asym25_checkpoint.py \
  --checkpoint-dir /ABS/CHECKPOINT_ROOT/transformer \
  --output evidence/contact_asym25/reviewer/seed6666_step0.json \
  --reviewer-identity "INDEPENDENT_REVIEWER_ID" \
  --reviewer-attestation-nonce "UNIQUE_REVIEW_NONCE_STEP0" \
  --acceptance-run-id "UNIQUE_ACCEPTANCE_RUN_ID_STEP0" \
  --acceptance-run-nonce "UNIQUE_ACCEPTANCE_RUN_NONCE_STEP0" \
  --current-run-matched-report /ABS/RUN_EVIDENCE/matched_control_report.json
```

Reviewer acceptance v2 re-reads the checkpoint bytes and producer receipt and
binds the checkpoint bundle, split seed, training-run ID and nonce. Its
reviewer-attestation nonce, acceptance-run ID and acceptance-run nonce are
independently authored and unique for each of the nine seed/step acceptances.
It also re-opens the current run's three transport-variant trace files,
independently replays the matched-control reducer without overwriting the
producer report, and requires the replayed report bytes to bind the selected
training run/nonce, stable run contract, current training-data/preflight/window
receipts, topology and external acceptance. A historical gate report cannot
stand in for this current-run report. Acceptance v1 is rejected.

Use the same receipt algorithm to fingerprint a base model or transformer
checkpoint tree:

```bash
conda run -n gigaworld-policy python -B scripts/evaluate_contact_asym25_heldout.py \
  fingerprint-tree \
  --path outputs/run_seed6666/models/checkpoint_epoch_1_step_100/transformer
```

The per-seed checkpoint manifest is strict JSON with this shape and exactly
three checkpoint entries:

```json
{
  "manifest_version": "contact-asym25-heldout-checkpoints/v2",
  "dataset_id": "DATASET_ID",
  "production_preflight_receipt_sha256": "64_HEX",
  "split_manifest_sha256": "64_HEX",
  "heldout_window_manifest_sha256": "64_HEX",
  "base_model": {"path": "/ABS/BASE_MODEL", "bundle_sha256": "64_HEX"},
  "checkpoints": [
    {
      "split_seed": 6666,
      "step": 0,
      "checkpoint_path": "/ABS/CHECKPOINT_ROOT/transformer",
      "checkpoint_bundle_sha256": "64_HEX",
      "checkpoint_bundle_manifest_path": "/ABS/CHECKPOINT_ROOT/transformer/checkpoint_bundle_manifest.json",
      "checkpoint_bundle_manifest_sha256": "64_HEX",
      "checkpoint_training_receipt_path": "/ABS/CHECKPOINT_ROOT/transformer/checkpoint_training_receipt.json",
      "checkpoint_training_receipt_sha256": "64_HEX",
      "independent_acceptance_path": "/ABS/EXTERNAL_REVIEW/seed6666_step0.json",
      "independent_acceptance_sha256": "64_HEX"
    }
  ]
}
```

Run the model evaluator only after the manifest contains accepted step 0, 100,
and 200. Production inference precision is fixed to bfloat16; Q12/D22/B16 and
the robot25 action-group schema are receipt-bound. Ego grip metrics use the
separate Ego20 schema and only dimensions 18/19 under the future-active mask.
It emits strict metrics JSON, a metrics SHA-256 sidecar, and hashed additive
per-window CSV evidence. Raw planner probabilities are used only for Brier,
NLL, and soft timing MAE; all F1/accuracy/exact metrics use the action-visible
hard-ST semantic tensor.

```bash
conda run -n gigaworld-policy python -B scripts/evaluate_contact_asym25_heldout.py \
  evaluate \
  --preflight-receipt evidence/contact_asym25/preflight_seed6666.json \
  --window-manifest evidence/contact_asym25/windows_seed6666.json \
  --checkpoint-manifest evidence/contact_asym25/checkpoints_seed6666.json \
  --output-dir evidence/contact_asym25/eval_seed6666
```

Repeat independently for seeds 7777 and 8888. The offline aggregator loads no
model; it replays all three receipts and window manifests, verifies each hashed
per-window CSV, recomputes each per-seed metric block, and then reduces the
three sets of additive rows. It additionally re-opens all nine acceptance
files and requires distinct reviewer-attestation nonces and acceptance-run
IDs/nonces, three distinct training-run IDs/nonces, one experiment
family/variant and the same initial weights, training code, dataset source,
action schema, precision and window/evaluator contract.

```bash
conda run -n gigaworld-policy python -B scripts/evaluate_contact_asym25_heldout.py \
  aggregate \
  --result evidence/contact_asym25/eval_seed6666/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/eval_seed7777/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/eval_seed8888/contact_asym25_heldout_metrics.json \
  --output evidence/contact_asym25/aggregate_three_seeds.json
```

For the same seed's three matched transport variants, run the separate offline comparison. It
requires the same preflight, fixed-window, matched-control and experiment-family
receipts, initial weights, training code, dataset source, action schema and
precision, but distinct variant IDs and training-run IDs/nonces.

```bash
conda run -n gigaworld-policy python -B scripts/evaluate_contact_asym25_heldout.py \
  check-matched-variants \
  --result evidence/contact_asym25/soft_semantic/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/hard_st/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/hard_detached/contact_asym25_heldout_metrics.json \
  --output evidence/contact_asym25/matched_variant_check.json
```

After all seeds and transports pass their per-result checks, the final matrix
checker must consume exactly nine results. It re-opens all 27 external
checkpoint acceptances and rejects any reused training-run ID/nonce,
reviewer-attestation nonce, acceptance-run ID/nonce or acceptance SHA. These
five identity namespaces are pairwise disjoint globally; an acceptance cannot
alias a reviewer nonce to its own run identity or to the training run. It also
re-applies the per-transport three-seed and per-seed three-transport contracts,
including byte-identical initial weights within every matched seed. Checkpoint
training receipt v4 embeds the versioned cross-seed normalization schema. The
checker canonicalizes the entire receipt and replaces only the exact listed
checkpoint-artifact, run-identity and seed-derived JSON-pointer leaves. Dataset
identity/source tree, hardware/capture, model/base/initial weights, training
and evaluation code, optimizer/scheduler/precision, control config, schema and
transport definitions are never normalized; unknown future fields therefore
must match by default.

Checkpoint bundle manifest v3 accepts one exact root layout only:
`config.json` plus either the non-empty `diffusion_pytorch_model.bin` or the
non-empty `diffusion_pytorch_model.safetensors`. The selected representation
must match the reviewed `checkpoint_safe_serialization` training setting;
config-only, mixed-format, nested, orphan and extra files are rejected. The
trainer emits reviewed receipts only at steps 0, 100 and 200. Later ordinary
checkpoints may still be saved by a long run, but cannot accidentally enter the
production evidence chain.

```bash
conda run -n gigaworld-policy python -B scripts/evaluate_contact_asym25_heldout.py \
  check-production-matrix \
  --result evidence/contact_asym25/seed6666/soft_semantic/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/seed6666/hard_st/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/seed6666/hard_detached/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/seed7777/soft_semantic/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/seed7777/hard_st/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/seed7777/hard_detached/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/seed8888/soft_semantic/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/seed8888/hard_st/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/seed8888/hard_detached/contact_asym25_heldout_metrics.json \
  --output evidence/contact_asym25/production_matrix.json
```
