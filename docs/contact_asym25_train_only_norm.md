# Production asymmetric25 train-only normalization

The contact/asymmetric25 production config accepts one normalization artifact
per `GWP_SPLIT_SEED`. Whole-dataset statistics remain valid only for legacy or
audit use; they are rejected by the production preflight because validation or
test episodes would otherwise influence training normalization.

The selected norm JSON keeps the existing `norm_stats` values and metadata
consumed by `WALeRobotTransformsPretrain`. Its `metadata` object must also
contain these two fields:

```json
{
  "production_train_norm_source_receipt": {},
  "production_train_norm_source_receipt_sha256": "64 lowercase hex"
}
```

`production_train_norm_source_receipt` has the exact schema shown in
`configs/contact_asym25_train_norm_source_receipt.template.json`. The SHA is
the digest of strict canonical JSON (`sort_keys=true`, compact separators,
UTF-8, no NaN). The ordinary metadata fields `split_seed`,
`train_episode_ids`, `train_episode_ids_sha256`,
`production_split_receipt_sha256`, `split_manifest_sha256`,
`source_dataset_tree_receipt`, `action_schema_id`, `model_dim`,
`action_horizon`, `delta_mask`, `action_representation`, `boundary`,
`num_episodes`, `num_state_samples`, `num_action_samples`,
`dataset_info_sha256`, and `episode_metadata_sha256` must mirror the receipt.
`data_roots` must contain exactly the production dataset root. This deliberate
duplication keeps normal metadata directly auditable while the exact nested
receipt and canonical hash make omissions or substitutions fail closed.

## Authoritative construction order

1. Complete and review group metadata and the three-seed split manifest.
2. Select exactly one seed and take only its sorted `train` episode IDs.
3. Hash `meta/info.json`, authoritative episode metadata, and every data
   parquet with the source-tree scope
   `meta-info+episode-metadata+data-parquet/v1`.
4. Compute state stats from every frame of only the selected train episodes.
   Compute 48-step action stats with the existing clamp-within-episode behavior,
   again only for those starts. Thus `num_state_samples` is the sum of selected
   train episode lengths and `num_action_samples` is that value multiplied by
   48.
5. Embed the exact source receipt and its canonical SHA in norm metadata. Keep
   a different immutable output file for each seed.
6. Update the externally reviewed training gate with the raw SHA-256 of that
   seed's complete norm JSON. Do not reuse a gate or norm artifact for another
   seed.

The production preflight independently rebuilds the group split identity,
episode counts, and source-tree receipt from runtime bytes. It rejects an
artifact with all-dataset counts, another seed, changed train IDs, changed
parquet/metadata bytes, or a non-canonical receipt digest. The production config
also passes the norm file SHA into each worker's transform loader so runtime
bytes are rechecked.

The current whole-dataset `scripts/compute_wam_task_norm.py` mode must not be
used to author this production artifact. No production norm should be generated
while group metadata is incomplete or the external training gate remains
false; the checked-in template is deliberately non-executable and contains no
claimed evidence.
