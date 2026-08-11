# Matched-control hash collector v2

This collector is an opt-in scientific control for WAM contact-transport
ablations. It is not a training or hardware-deployment gate by itself. The
only implemented runner IDs are exactly `soft_semantic`, `hard_st`, and
`hard_detached`; missing, extra, or alias variants are rejected.

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
requires an explicit dataset ID, run ID, 16-128 character run nonce, one of
seeds 6666/7777/8888, actual `RANK`/`WORLD_SIZE`, and an output directory below
`GWP05_PROJECT_DIR`. With the opt-in variable absent,
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
source bytes, run ID, and nonce before creating its rank-local file. The
comparator emits
`world_size`, `rank_set`, and an exact `topology` object so the production
preflight can bind the report to that external receipt instead of allowing the
report to certify itself.

Contact-asym25 gate v2 uses the stricter current-run mode. Its three transport
variants must have globally distinct run IDs, run nonces, and external
acceptance SHAs. Besides the ordinary control hash, every trace carries a
reviewed stable run contract covering current preflight/training code/data
tree, initial/base weights, bf16 and action/contact schema, runtime package and
launcher topology, optimizer/scheduler/training configuration, and the
train-window receipt. The contract embeds the exact training config and both
its actual-seed hash and a stable hash formed by replacing only `seed` with
`__CONTACT_ASYM25_SPLIT_SEED__`. The comparator also normalizes exactly one model field,
`contact_semantic_transport_mode`, and requires the remaining model-config
hash to be identical. Thus an apparent match produced by changing any second
model or training field is rejected even if the six sampled hashes happen to
agree.

The production expectation receipt also fixes the report byte SHA-256, exact
variant-to-trace path/SHA mapping, trace-manifest SHA, global-hashes SHA, and
hash-schema SHA. The preflight verifies report bytes before consuming fields,
then reopens and hashes every trace. A copied old report, missing trace,
same-path trace replacement, or self-consistent but externally unapproved
manifest cannot pass.

The Stage1 gate v3 also binds a separate training-launch nonce from the
environment and an explicit UTC launch-admission deadline. This bounds stale
bundle reuse but is not a one-time ledger: the same fully bound evidence may
be reused before expiry only for the same Git/data/config and reviewer gate.
The full production preflight compares expected/report Git commit with the
current repository HEAD and expected/report collector SHA with the installed
source bytes even when capture is disabled.

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

All collector trace/report JSON is strict JSON and writers use
`allow_nan=False`. Non-standard NaN/Infinity values (including exponent
overflow such as `1e999`) and booleans masquerading as integer topology/seed
fields are rejected. Offline reports are created with `O_EXCL`, followed by
file and parent-directory `fsync`; an existing evidence path is never
overwritten. The reviewer replay mode performs no directory creation or file
write. Production performs a minimal canonical-decimal
rank/world check before full artifact IO; the disabled path performs no
collector-artifact IO and does not mutate config or RNG state.

The Q0, capacity-control, auxiliary-only, raw-latent, shuffled, delayed, and
oracle rows remain without a matched-control runner. The exact implemented
runner set is `soft_semantic`, `hard_st`, and `hard_detached`; label and
asymmetric25 scientific gates remain closed.

## Causal-main collector v3

`world_action_model/contact_causal_matched_control.py` is an independent v3
collector and does not enlarge the v2 exact set above. Its exact route set is
`aux_only_direct_q12 / equal_bit_unstructured_hard_st / hard_st_semantic`.
The config validator rejects unknown keys, legacy aliases, boolean seeds,
missing optimizer/scheduler state, reused run identity/nonce, `/tmp` outputs,
and any self-admission flag.

In addition to the old batch/timestep/noise evidence, v3 hashes contact
targets, masks, confidence and grouped eligible counts; action targets and
masks; T5 input; pre-forward RNG; initial parameter tensors; optimizer groups
and step-0 state; scheduler step-0 state; and byte-identical precursor receipt.
The reducer compares the complete stable contract, so an unlisted future field
must match by default. Route-local fields remain explicit and the three route
run IDs/nonces must be globally distinct. See
`docs/contact_causal_production_matrix.md` for the versioned CLI and current
No-Go boundary.
