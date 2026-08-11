# Production asymmetric25 train-only normalization

The contact/asymmetric25 production config accepts one normalization artifact
per `GWP_SPLIT_SEED`. Whole-dataset statistics remain valid only for legacy or
audit use; they are rejected by the production preflight because validation or
test episodes would otherwise influence training normalization.

The selected norm JSON keeps the existing `norm_stats` values and metadata
consumed by `WALeRobotTransformsPretrain`. Its `metadata` object must also
contain these four fields:

```json
{
  "production_train_norm_source_receipt": {},
  "production_train_norm_source_receipt_sha256": "64 lowercase hex",
  "production_train_raw_statistics": {},
  "production_train_raw_statistics_sha256": "64 lowercase hex"
}
```

`production_train_norm_source_receipt` has the exact schema shown in
`configs/contact_asym25_train_norm_source_receipt.template.json`. The SHA is
the digest of strict canonical JSON (`sort_keys=true`, compact separators,
UTF-8, no NaN). The ordinary metadata fields `split_seed`,
`train_episode_ids`, `train_episode_ids_sha256`,
`production_split_receipt_sha256`, `split_manifest_sha256`,
`source_dataset_tree_receipt`, `action_schema_id`, `physical_dim`, `model_dim`,
`action_horizon`, `delta_mask`, `action_representation`, `boundary`,
`padding_contract`, `grip_validity_contract`, `num_episodes`,
`num_state_samples`, `num_action_samples`,
`dataset_info_sha256`, and `episode_metadata_sha256` must mirror the receipt.
`data_roots` must contain exactly the production dataset root. This deliberate
duplication keeps normal metadata directly auditable while the exact nested
receipt and canonical hash make omissions or substitutions fail closed.

The raw-statistics receipt contains count, dimension, sum, raw sum-of-squares,
centered sum-of-squares, minimum and maximum for state and action. Mean and
population standard deviation are derived from those moments. It also records
the selected raw-row SHA, per-train-episode row counts, verified data-file
receipt SHA and exact numeric contract. Production quantiles are exact NumPy
linear q01/q99 values; reservoir or sampled quantiles are forbidden. Its
deliberately empty schema example is
`configs/contact_asym25_train_raw_statistics.template.json`.

## Authoritative construction order

1. Complete and review group metadata and the three-seed split manifest.
2. Select exactly one seed and take only its sorted `train` episode IDs.
3. Hash `meta/info.json`, authoritative episode metadata, and every data
   parquet with the source-tree scope
   `meta-info+episode-metadata+data-parquet/v1`.
4. Read each parquet from bytes returned by the no-follow verified reader.
   Require exact global index, episode index and frame coverage, finite 25-D
   robot state/action values, and zero-pad only dimensions 25--31.
5. Compute state stats from every frame of only the selected train episodes.
   Compute 48-step action stats with the existing clamp-within-episode behavior,
   again only for those starts. Thus `num_state_samples` is the sum of selected
   train episode lengths and `num_action_samples` is that value multiplied by
   48.
6. Embed the exact source/raw receipts and their canonical SHAs in norm
   metadata. Keep a different immutable output file for each seed.
7. Update the externally reviewed training gate with the raw SHA-256 of that
   seed's complete norm JSON. Do not reuse a gate or norm artifact for another
   seed.

The production preflight independently rebuilds the group split identity,
episode counts, source-tree receipt, every sufficient statistic, and every
mean/std/min/max/q01/q99 value from runtime verified train bytes. It rejects an
artifact with all-dataset counts, another seed, changed train IDs, changed
parquet/metadata bytes, a non-canonical receipt digest, or validation/test rows
injected into emitted statistics. The production config also passes the norm
file SHA into each worker's transform loader so runtime bytes are rechecked.

## Two-phase generation gate

`scripts/compute_contact_asym25_train_norm.py` is the only production authoring
path. First run it with `--prepare-generation-request`. This validates the full
reviewed sidecar/group split and writes a request with
`allow_norm_generation=false`; it does not compute or write norm values. A
reviewer inspects its seed, split/source-tree/code SHAs, then sets only
`allow_norm_generation=true`, `reviewed_by`, and `reviewed_at_utc`. Run the same
inputs with `--output` and `--generation-gate` to write a new norm. The generator
refuses overwrites, false gates, changed code/source/split, incomplete group
metadata, and empty train/val/test partitions.

The request embeds the full production split receipt and source-tree receipt,
and separately binds the inventory CSV, group-policy build config, and split
manifest SHAs. Reviewers can therefore inspect the selected train IDs rather
than accepting a checkpoint or norm file's self-reported seed.

The generic shape in
`configs/contact_asym25_norm_generation_gate.template.json` is deliberately
all-false. Review the per-run request emitted by the generator instead of
filling the generic template by hand.

This contract has exactly 25 physical dimensions: wrist dimensions 0--17,
left RH56 dimensions 18--23, and right OmniPicker dimension 24. All seven grip
dimensions are valid on every selected robot row; dimensions 25--31 are model
padding and must remain exact zero. It must never consume an Ego active mask.
Ego20 normalization remains on the separate op30/v3 path, where future-active
grip and padded-time validity have their own contract.

The current whole-dataset `scripts/compute_wam_task_norm.py` mode remains
legacy/audit-only and must not author this production artifact. The production
generator implements robot25 only; it does not claim Ego active-grip or padded
future validity semantics. No production norm is generated while group
metadata is incomplete or its generation gate remains false. Checked-in
templates are deliberately non-executable and contain no claimed evidence.
