# Ego Stage1 strict discrete-contact entry

## Status and scope

The production-gated entry is
`configs/giga_world_policy_0_5_ego_discrete_contact_pretrain.py`. Importing it
runs a fail-closed preflight before the permissive base configuration is
loaded. It fixes 20 Hz, `processed_action_20hz`, H=48, Q=12/D=22,
`contact_segment_v1_1`, `hard_st`, B=16, strict semantic bottleneck, and
predicted contact conditioning. Label sidecars are training-only; action-only
inference predicts and caches the transported hard semantic prefix.

`configs/giga_world_policy_0_5_ego_contact_pretrain.py` remains intentionally
compatible with soft transport and the v0 54-D schema. It is legacy smoke
only: its schema, Q and bottleneck can be changed by environment variables; it
does not require reviewed annotation, provenance hashes, production group
splits, or explicit grip-polarity conversion. It must not support a scientific
strategy claim.

## Audited 66-episode source

The current processed LeRobot root reports 66 episodes, 20013 frames and 20
Hz. Its authoritative `meta/episodes.jsonl` covers episode IDs 0 through 65.
The 66 matching `vla_processed/final_hdf5/episode_*.hdf5` files contain exactly
`action_eef[T,14]`, `action_delta_eef[T,12]` and `hand_status[T,2]`, and all T
values match the LeRobot episode metadata. The earlier `hdf5/episode_*.hdf5`
files are 100 Hz controller/body streams and are not valid norm inputs.
The selected `final_hdf5` bundle is 66 files / 1,064,944,203 bytes and sits
outside the LeRobot root; it is a norm-only source with its own content bundle
hash, not part of the 4,321,599,931-byte training parquet receipt.
Across all 20013 audited rows, both hand-status columns are finite and the
only observed values are exactly 0 and 1.

The dataset has no source-authored `dataset_id` or `frame_clock`. The preflight
does not add or invent either. Instead,
`world_action_model/ego_dataset_fingerprint.py` derives a v2 portable identity
from:

- canonical stable fields of `meta/info.json`, including feature schema, fps,
  totals and split, while excluding `data_path` and `video_path`;
- canonical episode rows sorted by `episode_index`, retaining only episode ID,
  length and sorted task strings;
- a byte-verified training-source receipt over the exact files selected by the
  loader: for this v2.1 source, `meta/info.json`, `meta/tasks.jsonl`,
  `meta/episodes.jsonl`, optional `meta/stats.json`, every
  `data/*/*.parquet`, and only declared external video features. The portable
  `info.json` entry is canonicalized without its path fields; every other entry
  records a stable logical role/path, raw byte SHA-256 and byte size;
- the explicit derived Stage1 clock contract `processed_action_20hz`.

Canonical JSON uses sorted keys and compact separators, then SHA-256. Absolute
mount paths and mtimes do not enter the ID. Other loader-selected metadata
bytes and every parquet/video byte do enter it. Production deliberately accepts
only the audited LeRobot v2.1 layout (`info.json`, `tasks.jsonl`,
`episodes.jsonl`, optional `stats.json`); other metadata layouts fail closed.
The preflight recomputes the receipt rather than trusting a self-reported
manifest.

The actual call chain is
`WAMLeRobotDataset.open` -> `_LeRobotDatasetMetadata` ->
`FastLeRobotDataset.load_hf_dataset`/`__getitem__`. The compatibility metadata
loader reads the three v2.1 metadata files above; `load_hf_dataset` ignores
`info.data_path` and hard-codes `root/data/*/*.parquet`; `__getitem__` decodes
MP4 only when `meta.video_keys` is non-empty. Here the only visual feature,
`observation.images.left`, has dtype `image`, and the parquet Arrow schema is
`struct<bytes: binary, path: string>` (the first audited row contains 144553
embedded bytes). At dataset initialization the verified loader scans every
selected image chunk and rejects null/empty `bytes`, so a path-only image can
never become an unreceipted external read. No raw MP4 is read.
`meta/episodes_stats.jsonl` is not on this loader path and is intentionally
excluded.

The full 66-parquet audit covers all 20,013 image rows: 20,013 have non-empty
embedded bytes, 0 are path-only, 0 are empty, and the embedded image payloads
sum to 4,317,147,474 bytes. The adjacent raw collection root contains 67 MP4
files / 310,738,262 bytes, but none is referenced by the v2.1 loader schema;
they are therefore neither training inputs nor silently counted as the 66-file
norm HDF5 bundle.

The resulting real-source receipt covers exactly 69 files and
4,321,599,931 bytes: 3 metadata files / 9,084 bytes, 66 parquet files /
4,321,590,847 bytes, and 0 video files / 0 bytes.

```text
source_manifest_sha256: a4c64f51984c2eb2920afc715c52ebe842ce54d8f2a8804cff180d86c6ef50e9
source_receipt_sha256: d4873e6c0670a07aed9d48e4fea064ce56b402a408fff241bf6c3d243278d552
```

The resulting v2 dataset ID is:

```text
ego-processed-sha256:b5308e8a6882e128f324e097c41e817ad1a5230609aa0e20edce8caba2aebd5d
```

Every reviewed annotation build config must bind exactly this derived ID for
this data revision. Any semantic metadata or episode change produces a new ID.
The norm records and preflight checks the fingerprint algorithm version,
canonical info/episode field lists, episode sort key, task sort rule and JSON
canonicalization rule, so later readers cannot silently reinterpret the hash.

T5 is selected by dataloader config, not by LeRobot `info.json`, so it has a
separate production receipt instead of being confused with the portable
LeRobot dataset ID. Stage1 fixes `t5_load_from=dir` and hashes exactly one
`episode_{episode_index:06d}.pt` per audited episode. The manifest stores only
stable logical role + episode + content SHA/size; its absolute source directory
is record-only. The production gate independently binds both T5 manifest and
receipt SHA. The audited LeRobot root currently has no T5 files, so it remains
fail-closed until a complete 66-file T5 source is explicitly supplied.

## Action and norm contract

The source 20-D action is two 9-D wrist deltas plus two weak hand-state
proxies. Upstream `hand_status` is explicitly `0=open, 1=closed`; the transform
converts it to canonical `0=closed, 1=open` before constructing state/action.
The last two dimensions retain weight 0.25 because they are square-wave
proxies, not device aperture or finger contact. Weight 1.0 is a separate
ablation: the norm builder requires an explicit `--ablation-id`, changes the
action-schema identity, and the production preflight rejects it.

`scripts/compute_ego_wam_norm.py` requires both the authoritative LeRobot root
and the matching `final_hdf5` root. Its v2 metadata binds the dataset
fingerprint, full source manifest/receipt, canonical episode metadata,
source/target grip semantics, 20-to-32 zero padding, dimension mask, time-mask
policy, 0.25 grip weight, each HDF5 SHA-256, and a bundle SHA-256. The preflight
recomputes LeRobot, T5, and HDF5 hashes, so same-length content replacement is
rejected. Absolute T5/HDF5 paths are audit records only and do not affect their
portable receipts.

## Sidecar and evidence gates

Production accepts only builder-generated, human-adjudicated, reviewed v1.1
sidecars. It recomputes all windows from adjudicated CSVs and verifies exact
arrays, field masks, tau/censor constraints, Q=12 overflow policy, per-NPZ
hashes, physical-segment provenance, and bundle hash. It then requires exact
episode ID/length, 20 Hz and clock equality with the dataset contract.

The fixed seeds are 6666/7777/8888. Every seed must contain non-empty
train/val/test groups with no leakage. Subject/session/operator/source object/
target object/liquid/batch metadata must be complete and real. Episode-only
diagnostic splits, `ood_eligible=false`, `strategy_claim_allowed=false`, v0
sidecars, proposal labels and the episode-0 prescreen are hard failures.

The checked-in templates are deliberately all false/null. Authorization also
requires artifact-bound A/B agreement, Ego action/grip evidence, matched-control
collector evidence, and separate `allow_server_training` and
`allow_5b_training` decisions. No true gate example is provided.

Gate v3 additionally requires a reviewer-authored
`evidence_valid_until_utc` and a 16-128 character `training_run_nonce` that
must exactly equal `GWP_EGO_TRAINING_RUN_NONCE`. Expiry is a launch-admission
deadline: it does not terminate an already admitted run. This is bounded
freshness plus explicit launch binding, not a one-time-consumption ledger;
reuse inside the deadline is allowed only when Git/data/config, the external
gate, and the explicit nonce are all unchanged.

The matched-control report must expose exactly one `dataset_id`. Both its
scalar `dataset_id` and singleton `dataset_ids` list must equal the portable ID
derived by the Stage1 preflight. The production gate must independently state
the expected full Git commit, control-config SHA256, complete variant-to-config
SHA256 mapping, world size, complete ordered rank set, collector version, and
collector source SHA256. The preflight compares every report field against
those reviewer-authored values; it never copies expectations out of the report
being checked. It additionally binds a reviewer-selected run ID and nonce, the
report file SHA-256, exact trace path/SHA set, and the digests of global hashes
and hash schema. Every trace is reopened and hashed. Missing, replayed, or
tampered artifacts fail. Null, partial, stale, or self-described expectations
fail.

The full preflight performs the current-HEAD and installed collector-source
checks regardless of whether in-process collection is enabled. Every trusted
JSON/JSONL is strict JSON (NaN, Infinity, `1e999`, strings and booleans cannot
stand in for agreement numbers), and agreement values are checked for both
physical range and threshold. Gate/report JSON, evidence CSVs, sidecar CSVs,
build config, split and provenance are each parsed from the same no-follow
bytes whose digest is checked.

Optional in-process capture receives the portable dataset identity and that
same expectation receipt from the preflight. On the enabled path it recomputes
the effective hard-ST config hash, current Git commit and collector source hash,
then checks runtime rank topology before creating a trace file. An environment
value cannot replace the dataset receipt. `RANK` and `WORLD_SIZE` are parsed
without initializing a collective or GPU and must match the external topology
before any config mutation or path creation. With
`GWP_MATCHED_CONTROL_OUTPUT_DIR` unset, collector wiring returns before reading
other collector variables, hashing files, resolving Git, mutating config,
creating paths, or consuming RNG.

When collection is enabled, a stdlib-only topology bootstrap runs before the
full preflight and before importing the legacy config. It reads only the
explicit gate JSON plus canonical ASCII-decimal `RANK`/`WORLD_SIZE`; a mismatch
therefore fails before dataset/T5/norm/report hashing or config mutation. The
disabled bootstrap performs no collector-artifact access.

The full preflight passes record-only runtime receipts (absolute path plus
SHA-256 and device/inode/mode/size/mtime/ctime) to each rank. Metadata,
parquet, and video files are opened with no-follow and pinned by FD. HF parquet
materialization reads those pinned descriptors; its content-keyed Arrow cache
is force-built on first use, fully attested on reuse, then pinned too. Cache
FDs are stat-checked immediately before and after each training item read, so
in-place cache mutation fails. Fork workers inherit already verified FDs and
perform one stat verification; spawn workers serialize no live FD and safely
rehash/reopen once. There is no 69-file fstat loop per sample. T5, reviewed
NPZ sidecars, and norm JSON are loaded from the exact verified bytes via
`BytesIO`/in-memory JSON, so pathname replacement after verification cannot
change the consumed payload.

The cost is explicit: each rank streams SHA-256 over the 4.32 GB LeRobot
source during full preflight and again when the actual loader pins it, plus the
content-keyed Arrow cache on first admission. A spawned worker must reverify
once; fork workers do not rehash. T5/sidecar caches and all receipt trust are
process-local. Pinned source descriptors use a deterministic range beginning
at FD 16384 so the Hugging Face cache key cannot drift with incidental open-FD
order; production therefore requires a per-process `RLIMIT_NOFILE` above the
highest selected source FD (16452 for the audited 69-file source) and fails
closed if that range is unavailable. Runtime stat records never enter the
portable dataset ID or matched-control config hash.

## Current decision

Current data are **No-Go for production Stage1**. The 66-episode inventory lacks
required subject/session/operator/object/liquid/batch metadata, the existing
split is diagnostic and strategy-ineligible, and episode 0 is only a proposal.
The new entry therefore cannot be imported successfully with current Track A
artifacts. Legacy smoke remains available for CPU/interface checks and does not
upgrade those artifacts to ground truth.
