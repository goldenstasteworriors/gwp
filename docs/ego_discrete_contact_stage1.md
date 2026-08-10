# Ego Stage1 strict discrete-contact entry

## Status and scope

The production-gated entry is
`configs/giga_world_policy_0_5_ego_discrete_contact_pretrain.py`. Importing it
runs a fail-closed preflight before the permissive base configuration is
loaded. A stdlib-only admission pass rejects missing required paths, false or
stale reviewer gates, nonce mismatch, and a missing/symlinked T5 directory
before NumPy, Torch, dataset, or model imports. This pass can only reject; the
full preflight reopens and verifies every artifact before config construction.
It fixes 20 Hz, `processed_action_20hz`, H=48, Q=12/D=22,
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
LeRobot dataset ID. Stage1 fixes `t5_load_from=dir` and requires exactly one
`episode_{episode_index:06d}.pt` per runtime-audited episode plus
`t5_source_manifest.json`. The latter binds the runtime-derived dataset ID,
episode metadata digest, every exact prompt, model repository and full
revision, checkpoint SHA/Hugging Face etag, tokenizer files, WAN encoder code,
text cleaning/tokenization settings, compute/output dtype, current-HEAD code
blobs plus installed code bytes, file SHA and logical tensor SHA. This manifest
is explicitly a **generator-authored generation claim**, not independent proof
that those assets produced the tensors. The production receipt includes the
claim manifest; its absolute source directory remains record-only.

The local `b530...` LeRobot root currently has zero T5 files. A separate
read-only 4x server snapshot contains 66 legacy files, each a finite float32
`[11,4096]` tensor. All 66 logical tensor payloads are identical, which is
expected because all 66 metadata rows contain the same exact prompt,
`pour liquid from one beaker into another beaker`. Those files nevertheless
lack model/tokenizer/generator provenance and belong to a different
runtime-derived dataset receipt (its metadata was previously augmented with
T5 path pointers). They are **not** portable to the local `b530...` source and
remain No-Go. The strict generator never modifies `episodes.jsonl`, never adds
T5 pointers, and refuses an existing output directory, so legacy files cannot
be upgraded merely by writing a JSON claim beside them.

The legacy files and the pointer-augmented `episodes.jsonl` were written within
the same second, immediately after the cache/dedup changes to
`scripts/compute_t5_embedding.py`. That is strong evidence that the old script
produced them, but it is not proof of the command line, checkpoint revision,
tokenizer bytes, device/dtype, or source commit used at execution time; the
production decision therefore remains unchanged. Local Hugging Face download
metadata is not a trust root: generation requires reviewer-supplied checkpoint
SHA, model revision and HF etag, then checks the actual checkpoint bytes.

The production loader call chain is fixed to `t5_load_from=dir`:
`WAMLeRobotDataset` resolves `episode_{episode_index:06d}.pt`, verifies the
preflight SHA/stat receipt, uses `torch.load(..., weights_only=True)`, and
accepts only a direct finite float32 tensor with shape `[1..64,4096]`.
`WALeRobotTransformsPretrain` keeps at most 64 rows and zero-pads to
`[64,4096]` before the 4096-D text projection. The legacy `pkl` main/subtask
lookup is not in this path, and Stage1 fixes `subtask_prob=0`; no unbound
subtask prompt can be sampled. WAN tokenization is fixed to whitespace
cleaning, `text_len=512`, padding/truncation at tokenizer input, and
`add_special_tokens=True`; the encoder output is sliced back to the non-padding
token count before the Stage1 64-token contract is enforced.

WAN checkpoint loading is single-open and fail-closed: `O_NOFOLLOW` pins one
regular-file descriptor, the same FD is stream-hashed, rewound and passed as a
file object to `torch.load(..., weights_only=True, map_location="cpu")`, only a
non-empty string-key tensor state dict is accepted, and descriptor/path
identity is checked again after loading. The four tokenizer files are copied
from verified bytes into an owner-only, no-extra-file staging snapshot;
`AutoTokenizer` uses only that snapshot with `local_files_only=True`, and every
snapshot file is rehashed with its original identity after tokenizer loading.
The hard-coded UMT5 model configuration is bound by the WAN encoder code hash.
Generator/WAN code pins one HEAD commit, directly reads every `HEAD:path` blob,
and requires its byte length and SHA-256 to equal the separately no-follow
runtime file read. This equality is rechecked before and after importing the
compute child, loading the encoder and every unique-prompt encoding; preflight
performs the same direct-blob comparison. Generation and replay reject a WAN
code path that is not the current checkout's `third_party` directory.

Both generator and replay must be launched as their script file with
`python -B`. Before importing any project or WAN module, the main script
normalizes `sys.path` to one checkout root, rejects another path that could
resolve the same project packages, rejects existing `__pycache__` directories
or `.pyc` files below the code roots, creates a new empty owner-only pycache
prefix on the project mount, and points `sys.pycache_prefix` at it. `-B` stops
writes but is not treated as sufficient to prevent cache reads; the empty
prefix provides that isolation. Namespace imports from `scripts` are forbidden:
prepare/compute/WAN helpers are compiled and executed from the exact verified
HEAD-equal source bytes, and their `__file__` plus `spec.origin` must identify
the manifest path. The main script itself is executed from its source path;
the selected Python interpreter, standard-library startup, `sitecustomize` and
third-party site-packages remain environment-receipt boundaries. Arbitrary
same-UID mutation of the running process or restoration races remains outside
the boundary and requires UID or mount-namespace isolation.

Run these commands from a clean, dedicated worktree with no project-code
`__pycache__`, `.pyc`, or stale `.gwp-secure-pycache-*` entry. The tools only
fail closed on pre-existing caches and never remove them; they clean up only
the exact private prefix created by their own current process.

A read-only 4x trace on the actual tokenizer directory used Transformers
4.51.3 and Tokenizers 0.21.4 with all framework backends disabled. Linux
inotify observed `T5TokenizerFast` open/read only `tokenizer_config.json` and
`tokenizer.json`; it did not access `config.json` or `generation_config.json`.
The snapshot still binds all four audited tokenizer files, including
`special_tokens_map.json` and `spiece.model`, as a conservative exact inventory.
Both Transformers and Tokenizers versions are recorded in generation and
independent-replay evidence.

All staging directories and files use exclusive creation. Files and directories
are fsynced before Linux `renameat2(RENAME_NOREPLACE)` publication, followed by
a parent-directory fsync; a concurrent target or unsupported atomic primitive
fails closed. These controls cover ordinary filesystem races and crashes. Code
running under the same UID can attack process memory or forge metadata and is
outside this boundary; production requiring that threat model needs a distinct
UID or mount namespace.

Prompt planning is CPU-only and does not import or probe an accelerator:

```bash
conda run -n gigaworld-policy python -B scripts/prepare_ego_t5_source.py plan \
  --data-root /home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers/vla_processed/lerobot/7_4_pour_liquid_between_beakers
```

Future generation on `4xH20-node2` must write a new directory on the same
`/workspace` mount. The audited 4x model asset is the WAN 2.2 TI2V 5B UMT5
checkpoint at revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`; generation
rehashes the 11.36 GB checkpoint, tokenizer and encoder code before loading the
model. Because this dataset has only one unique prompt, the encoder runs once
and the result is copied into 66 independently hashed episode files:

First verify the project runtime without loading the 11.36 GB checkpoint or
running an accelerator forward:

```bash
python -B scripts/prepare_ego_t5_source.py runtime-check \
  --wan-code-path /workspace/WM/giga-world-policy-contact-transition/third_party \
  --device cuda:0
```

This gate requires importable `torch`, `transformers`, `ftfy`, `regex`, the WAN
encoder class, and an available requested device. Their exact versions are
recorded in the generated manifest. The local README-named conda environment
currently lacks torch, while the inspected EgoHumanoid virtual environment has
torch but lacks `ftfy`; neither local environment is generation-ready, and no
dependency was installed during this audit. No project-owned Python/conda
runtime was found under the inspected 4x `/workspace` checkout, and its system
Python cannot currently import torch because a vendor runtime library is
missing. The 4x runtime-check is therefore also presently No-Go. Resolve this
only through the vendor-supported project environment; do not change the
device driver or system CUDA/runtime stack for this T5 task.

```bash
python -B scripts/prepare_ego_t5_source.py generate \
  --data-root /workspace/WM/giga-world-policy-contact-transition/data/7_4_pour_liquid_between_beakers \
  --output-dir /workspace/WM/giga-world-policy-contact-transition/data/7_4_pour_liquid_between_beakers/t5_embedding_verified_v1 \
  --model-root /workspace/WM/giga-world-policy/weights/Wan2.2-TI2V-5B-Diffusers \
  --wan-code-path /workspace/WM/giga-world-policy-contact-transition/third_party \
  --model-revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --checkpoint-sha256 "$REVIEWER_CHECKPOINT_SHA256" \
  --checkpoint-hf-etag "$REVIEWER_CHECKPOINT_HF_ETAG" \
  --device cuda:0
```

The command above is a future execution plan, not an executed result. It must
use the project environment and the vendor-supported accelerator stack; it
must not change CUDA or driver versions. One UMT5-XXL load plus one short
forward is expected to take a few minutes on one accelerator, with roughly
the checkpoint/model footprint (about 12 GB on disk and materially more device
memory during load). CPU float32 generation avoids accelerator use but expands
the model/state memory requirement to tens of GB and is expected to be much
slower. In either mode the saved T5 sidecar itself is only about 12 MB.

After generation, ordinary validation can rehash model assets, prompts and
tensor bytes, but it remains in the generator trust domain and cannot unlock
the training gate:

```bash
python -B scripts/prepare_ego_t5_source.py validate \
  --data-root /workspace/WM/giga-world-policy-contact-transition/data/7_4_pour_liquid_between_beakers \
  --t5-embedding-dir /workspace/WM/giga-world-policy-contact-transition/data/7_4_pour_liquid_between_beakers/t5_embedding_verified_v1 \
  --model-root /workspace/WM/giga-world-policy/weights/Wan2.2-TI2V-5B-Diffusers \
  --wan-code-path /workspace/WM/giga-world-policy-contact-transition/third_party \
  --model-revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e
```

In a separate process, with a new replay output directory and a previously
nonexistent attestation path, replay every unique prompt and compare exact
float32 contiguous tensor bytes:

```bash
python -B scripts/verify_ego_t5_replay.py \
  --data-root /workspace/WM/giga-world-policy-contact-transition/data/7_4_pour_liquid_between_beakers \
  --t5-embedding-dir /workspace/WM/giga-world-policy-contact-transition/data/7_4_pour_liquid_between_beakers/t5_embedding_verified_v1 \
  --replay-output-dir /workspace/WM/giga-world-policy-contact-transition/data/7_4_pour_liquid_between_beakers/t5_embedding_replay_v1 \
  --attestation-output /workspace/WM/giga-world-policy-contact-transition/evidence/t5_independent_replay_v1.json \
  --model-root /workspace/WM/giga-world-policy/weights/Wan2.2-TI2V-5B-Diffusers \
  --wan-code-path /workspace/WM/giga-world-policy-contact-transition/third_party \
  --model-revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --checkpoint-sha256 "$REVIEWER_CHECKPOINT_SHA256" \
  --checkpoint-hf-etag "$REVIEWER_CHECKPOINT_HF_ETAG" \
  --device cuda:0
```

The verifier has no gate argument and cannot modify the source or gate. A
reviewer must separately place the attestation raw SHA-256 in both the v5
gate's `artifact_fingerprints.t5_independent_replay_attestation_sha256` and
`evidence_reports.t5_independent_replay.sha256`, plus its path. The checked-in
template keeps these values null and every authorization flag false. A v3 gate,
a self-authored/fake attestation, or an attestation missing any unique prompt is
rejected.

## Action and norm contract

The source 20-D action is two 9-D wrist deltas plus two weak hand-state
proxies. Upstream `hand_status` is explicitly `0=open, 1=closed`; the transform
converts it to canonical `0=closed, 1=open` before constructing state/action.
The last two dimensions retain weight 0.25 because they are square-wave
proxies, not device aperture or finger contact. Weight 1.0 is a separate
ablation: the norm builder requires an explicit `--ablation-id`, changes the
action-schema identity, and the production preflight rejects it.

The source tracker-validity fields are separate, scalar
`left_hand_active`/`right_hand_active` arrays with `0=inactive, 1=active`.
`scripts/build_ego_hand_active_sidecars.py` verifies the exact raw-100-Hz to
processed-20-Hz timestamp relation (`raw[::5]`), fixes field order to
`[left,right]`, and writes H=48 masks. For a source frame `f`, future target
slot `k` uses validity at `f+k+1`; targets beyond the episode boundary are
false rather than last-value held. Its record-only transition mask requires
validity at both target-1 and target. No transition auxiliary is enabled.

For the production path, action dimensions 18/19 receive weight
`0.25 * active_at_target` independently for the left/right columns. Dimensions
0..17 remain unchanged, 20..31 remain padding, and the action-time mask also
zeros clamped horizon padding. Without the optional validity keys the legacy
transform behavior is unchanged (0.25 on both grip dimensions); the gated Ego
entry always configures the strict sidecars. `hand_active` is tracker validity,
not physical-contact truth.

The audited 66 episodes contain 99,953 raw frames and 20,013 exact-stride
processed frames. Raw active counts are 35,445 left and 32,877 right; 20-Hz
current active counts are 7,087 left and 6,574 right. Across all valid H=48
future targets there are 335,074 left and 308,116 right grip supervision
samples. The zero-active episode counts must not be mixed: raw-100-Hz has 28
left/29 right, exact-stride current-20-Hz has 30 left/29 right, and H=48 future
targets have 30 left/29 right. Left episodes 15 (raw indices 418,419) and 48
(512,513,514) lose their only raw active samples because none lies on an index
divisible by five.

`scripts/compute_ego_wam_norm.py` requires the authoritative LeRobot root, the
matching raw/final HDF5 roots, and the verified validity sidecar bundle. Its v3
metadata binds the dataset
fingerprint, full source manifest/receipt, canonical episode metadata,
source/target grip semantics, validity statistics, raw/final HDF5 and NPZ
bundle hashes, 20-to-32 zero padding, dimension mask, time-mask policy, 0.25
active grip weight, each HDF5 SHA-256, and a bundle SHA-256. Grip dimensions
18/19 are normalized only from their corresponding active samples; inactive
held proxy values and clamped targets are excluded. The preflight independently
replays these masked grip statistics and recomputes LeRobot, T5, HDF5, and
sidecar hashes, so same-length content replacement is rejected. Absolute
T5/HDF5 paths are audit records only and do not affect their portable receipts.

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

Gate v5 additionally requires a reviewer-authored
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
JSON/JSONL is strict JSON (duplicate keys, NaN, Infinity, `1e999`, strings and booleans cannot
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
parquet, and video files are opened with no-follow and pinned by dynamically
allocated FDs. HF parquet materialization reads those pinned descriptors. Its
builder identity is derived from source/config content and is independent of
incidental FD numbers.

Verified production loading never accepts a shared HF cache or a colocated
cache attestation. Before every actual reader process builds Arrow, it creates
an unpredictable run-nonce/rank/PID/random leaf below the checked-in project's
private-cache parent using exclusive `mkdirat`, pins both parent and leaf by
directory FD, fixes the leaf to owner-only mode 0700, and gives HF the pinned
`/proc/self/fd/<dirfd>` root. It always uses `FORCE_REDOWNLOAD`; existing cache
bytes and self-authored manifests have no admission path. The generated Arrow
files are opened through component-wise no-follow `openat`, hashed and pinned
before Dataset creation. `datasets.Dataset.from_file` receives only the
corresponding `/proc/self/fd/<arrow-fd>` paths, so its mmap inode is the exact
inode checked at runtime rather than a later pathname reopen.

Before and after each item, the loader checks both the actual Arrow mmap source
FD and its no-follow directory entry, as well as the private parent/leaf
binding. In-place mutation, file replacement, directory rename, or symlink
replacement therefore fails closed. `WAMLeRobotDataset.__len__` reads only the
verified metadata receipt, so it does not build a cache in the parent before
DataLoader workers are created. A fork/spawn worker discards inherited mmap and
cache descriptors without deleting its parent's directory, then creates and
force-builds its own private cache. Pickled state contains no mmap or live FD.
Only a process that still owns and verifies the leaf binding removes that leaf.
T5, reviewed NPZ sidecars, and norm JSON remain exact verified
`BytesIO`/in-memory payloads.

The cost is explicit: full preflight hashes the 4.32 GB LeRobot source, and
each actual reader process rebuilds its own private Arrow cache; spawn readers
reopen and hash the source, while fork readers retain the already verified
source FDs and recheck their stat identity. There is deliberately no
persistent-cache speed path.
Per-item checks scale with the Arrow shards actually mmaped (and the video file
actually decoded), not all 69 training-source files. FD allocation is dynamic:
low limits are accepted when sufficient for the files actually held, while
real `EMFILE`/`ENFILE` exhaustion fails closed without a reserved-FD range.
Runtime cache paths, nonce and stat records never enter the portable dataset ID
or matched-control config hash. Mode 0700 isolates other UIDs; deployments that
treat arbitrary code under the training UID as hostile still require an OS
boundary such as a distinct UID or mount namespace.

FD acquisition, adoption, multi-FD transfer, publication, and consume-close
transitions use short POSIX per-thread `pthread_sigmask` critical sections and
restore the exact prior mask in `finally`; they never replace the process-wide
SIGINT handler. Ordinary exceptions retain caller/owner cleanup. This guarantee
applies to SIGINT targeted at the participating thread, and to process-directed
SIGINT when all other threads that can receive it have inherited or installed
the same block. POSIX may deliver a process-directed SIGINT to an already
existing unblocked thread; this implementation does not claim to defer that
case. Python executes its signal handler on the main thread, and termination
reclaims any process FDs; the limitation is an availability boundary, not an
authorization for consuming unverified bytes or changing the pinned inode.
Synthetic `sys.settrace` exceptions inside a signal-blocked transfer are not a
production admission criterion; explicit function exceptions still exercise
the RAII cleanup paths.

## Current decision

Current data are **No-Go for production Stage1**. No real independent replay
attestation or reviewer-authored v4 gate exists. The 66-episode inventory lacks
required subject/session/operator/object/liquid/batch metadata, the existing
split is diagnostic and strategy-ineligible, and episode 0 is only a proposal.
The new entry therefore cannot be imported successfully with current Track A
artifacts. Legacy smoke remains available for CPU/interface checks and does not
upgrade those artifacts to ground truth.
