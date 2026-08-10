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
Across all 20013 audited rows, both hand-status columns are finite and the
only observed values are exactly 0 and 1.

The dataset has no source-authored `dataset_id` or `frame_clock`. The preflight
does not add or invent either. Instead,
`world_action_model/ego_dataset_fingerprint.py` derives a portable identity
from:

- canonical stable fields of `meta/info.json`, including feature schema, fps,
  totals, split and relative data template;
- canonical episode rows sorted by `episode_index`, retaining only episode ID,
  length and sorted task strings;
- the explicit derived Stage1 clock contract `processed_action_20hz`.

Canonical JSON uses sorted keys and compact separators, then SHA-256. Absolute
paths, mtimes and JSON/parquet formatting do not enter the ID. Both JSONL and
v3.1-style parquet episode metadata are supported. For the audited source the
derived ID is:

```text
ego-processed-sha256:bebcf2575b34d78816357c33d458f1c4443cbaf77e6d35fd68904e9f5ec2f990
```

Every reviewed annotation build config must bind exactly this derived ID for
this data revision. Any semantic metadata or episode change produces a new ID.
The norm records and preflight checks the fingerprint algorithm version,
canonical info/episode field lists, episode sort key, task sort rule and JSON
canonicalization rule, so later readers cannot silently reinterpret the hash.

## Action and norm contract

The source 20-D action is two 9-D wrist deltas plus two weak hand-state
proxies. Upstream `hand_status` is explicitly `0=open, 1=closed`; the transform
converts it to canonical `0=closed, 1=open` before constructing state/action.
The last two dimensions retain weight 0.25 because they are square-wave
proxies, not device aperture or finger contact. Weight 1.0 is a separate
ablation: the norm builder requires an explicit `--ablation-id`, changes the
action-schema identity, and the production preflight rejects it.

`scripts/compute_ego_wam_norm.py` requires both the authoritative LeRobot root
and the matching `final_hdf5` root. Its metadata binds the dataset fingerprint,
canonical episode metadata, source/target grip semantics, 20-to-32 zero
padding, dimension mask, time-mask policy, 0.25 grip weight, each HDF5 SHA-256,
and a bundle SHA-256. The preflight recomputes the HDF5 hashes, so a same-length
action replacement is rejected. Absolute HDF5 paths are audit records only and
do not affect the portable dataset identity.

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

## Current decision

Current data are **No-Go for production Stage1**. The 66-episode inventory lacks
required subject/session/operator/object/liquid/batch metadata, the existing
split is diagnostic and strategy-ineligible, and episode 0 is only a proposal.
The new entry therefore cannot be imported successfully with current Track A
artifacts. Legacy smoke remains available for CPU/interface checks and does not
upgrade those artifacts to ground truth.
