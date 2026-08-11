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
before it opens any samples, and embeds the current training-data receipt in
the transformer config and adjacent checkpoint binding file.

```bash
export GWP_CONTACT_ASYM25_PREFLIGHT_RECEIPT="$PWD/evidence/contact_asym25/preflight_seed6666.json"
export GWP_CONTACT_HELDOUT_WINDOW_MANIFEST="$PWD/evidence/contact_asym25/windows_seed6666.json"
```

Step 0 is saved before optimization. Periodic saving is fixed at 100 steps, so
step 100 and step 200 are present. The evaluator's `checkpoint_path` is the
saved transformer subdirectory (for example
`checkpoint_epoch_1_step_100/transformer`); the adjacent binding path is the
checkpoint-root `current_training_data_receipt.json`.

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
  "manifest_version": "contact-asym25-heldout-checkpoints/v1",
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
      "current_training_data_receipt_path": "/ABS/CHECKPOINT_ROOT/current_training_data_receipt.json",
      "current_training_data_receipt_sha256": "64_HEX"
    }
  ]
}
```

Run the model evaluator only after the manifest contains step 0, 100, and 200.
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
three sets of additive rows.

```bash
conda run -n gigaworld-policy python -B scripts/evaluate_contact_asym25_heldout.py \
  aggregate \
  --result evidence/contact_asym25/eval_seed6666/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/eval_seed7777/contact_asym25_heldout_metrics.json \
  --result evidence/contact_asym25/eval_seed8888/contact_asym25_heldout_metrics.json \
  --output evidence/contact_asym25/aggregate_three_seeds.json
```
