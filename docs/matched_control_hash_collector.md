# Matched-control hash collector v1

This collector is an opt-in scientific control for WAM contact-transport
ablations. It is not a training or hardware-deployment gate by itself.

## Observed values

The dataset supplies a stable identity containing `dataset_id`,
`episode_index`, `frame_index`, and dataset-local `global_index`. The absolute
`data_path` is diagnostic only and is excluded from the identity hash. Before
the transform makes any RNG call, it hashes the ordered raw image tensors and
the state/action source tensors, plus available prompt, contact, padding, and
loss-mask inputs. Tensor and ndarray serialization always includes dtype,
shape, and contiguous CPU raw bytes.

The transform returns a fixed-schema per-sample trace to the collator. It
records the already sampled augmentation values: crop boxes, rotation angle,
color-jitter factors and application order, final resize-crop, reference-mask
length, and subtask-prompt draw. DataLoader workers never write trace files.

The trainer records, after their existing sampling calls, visual/action
timesteps and visual/action noise. It writes one JSONL file per rank. No RNG
call and no distributed collective is added. The offline comparator orders
ranks numerically and constructs global hashes from the rank-local hashes.

## Opt-in contract

`configs/giga_world_policy_0_5_sonic_g1_discrete_contact_ablation.py` enables
capture only when `GWP_MATCHED_CONTROL_OUTPUT_DIR` is set. It then also
requires an explicit dataset ID, run ID, one of seeds 6666/7777/8888, and an
output directory below `GWP05_PROJECT_DIR`. With the opt-in variable absent,
the dataset, transform, trainer output, and RNG path remain on the legacy
route.

Production configs must call the following after their preflight:

```python
configure_matched_control_from_environment(
    ...,
    expected_dataset_id=receipt["dataset_id"],
    **receipt["matched_control_expectations"],
)
```

When capture is enabled, the helper rejects any
environment dataset ID that differs from this receipt-bound value. The
optional argument also supports portable derived IDs for legacy Ego metadata;
it does not require editing the source dataset. When capture is disabled, the
helper returns before reading any other collector environment variable and
does not mutate the config or create files.

Production Ego Stage1 additionally passes the complete reviewer-authored
expectation receipt: Git commit, control hash, variant hash mapping, world/rank
topology, collector version, and collector source hash. The enabled helper
recomputes all locally knowable values before mutating the config. Each runtime
collector independently checks its rank/world identity, Git commit, version,
and source bytes before creating its rank-local file. The comparator emits
`world_size`, `rank_set`, and an exact `topology` object so the production
preflight can bind the report to that external receipt instead of allowing the
report to certify itself.

The comparator rejects missing fields; duplicate runs, files, or record keys;
different seed/step/rank sets; incomplete ranks; different sample identities;
any of the six unequal hashes; different control-config hash, hash scope, Git
commit, collector version, or collector code SHA. Variant-config hashes are
expected to differ and are retained in the report. A passed report binds the
exact trace-file SHA256 values and cannot be generated from empty traces.
It recomputes the dataset IDs from every recorded `batch_ids` entry, requires
one dataset across all ranks, steps, seeds, and variants, and emits both the
scalar `dataset_id` and auditable `dataset_ids` list for the Stage-1/Stage-2
preflight binding.

The Q0, capacity-control, auxiliary-only, raw-latent, shuffled, delayed, and
oracle rows remain without a matched-control runner. The implemented runner
scope is `soft_semantic`, `hard_st`, and `hard_detached`; label and
asymmetric25 scientific gates remain closed.
