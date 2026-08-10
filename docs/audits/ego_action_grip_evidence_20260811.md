# Ego action/grip semantics and missingness audit (2026-08-11)

## Decision

The 66-episode `7_4_pour_liquid_between_beakers` source supports the reviewed
status `semantics_verified_missingness_masked`. It does **not** support a
physical-contact-truth claim. Production admission still requires the external
gate to supply all exact artifact hashes and reviewer authorizations.

## Semantics and alignment

- Upstream EgoHumanoid commit:
  `cb0c0dc06f4e1e90e125dfb7931827e4d8d125de`.
- `hand_status` is `0=open, 1=closed`, with columns `[left,right]`.
- Absolute EEF pose and local delta order is left then right. The processed
  delta is `delta[t] = inverse(pose[t-1]) * pose[t]`; `delta[0]` is zero.
- WAM converts the two hand columns to canonical open fractions by `1-source`.
  Canonical action dimensions 18 and 19 are left and right grip respectively.
- Processed time uses raw indices `[::5]`. The final HDF5 stores timestamps as
  `float64`; values equal the selected raw timestamps after this exact dtype
  cast. Relative to the original integer nanoseconds, maximum rounding is 128
  ns, equal to the half-ULP bound. It is not reported as zero-nanosecond
  identity.
- Independent numerical replay over all episodes found maximum translation
  residual `2.255140518769849e-16`, wrapped XYZ-Euler residual
  `4.440892098500626e-16`, quaternion norm error
  `2.220446049250313e-16`, and exact zero first deltas.

The audited upstream worktree differs from HEAD only for the checked
`add_hand_status.py` file. Both the HEAD blob and current worktree contain all
required semantics; both hashes and `worktree_matches_head=false` are recorded.
The historical HDF5 generator commit was not embedded in the dataset, so this
limitation remains explicit.

## Three distinct validity domains

`raw_100hz_active` means source tracker validity before downsampling.
`current_20hz_active` means validity at the exact `[::5]` processed frames.
`future_target_supervision` means current-20-Hz validity at target
`f+offset`, offsets 1..48; out-of-episode targets are false.

| Domain | Left active/count | Right active/count | Left zero episodes | Right zero episodes |
| --- | ---: | ---: | ---: | ---: |
| raw 100 Hz | 35,445 / 99,953 | 32,877 / 99,953 | 28 | 29 |
| current 20 Hz | 7,087 / 20,013 | 6,574 / 20,013 | 30 | 29 |
| valid H=48 future targets | 335,074 | 308,116 | 30 | 29 |

Left raw-zero set:

```text
1,3,5,7,11,13,17,19,21,23,25,27,29,31,33,35,37,44,46,49,50,52,54,56,58,60,62,64
```

Left current-20-Hz and future-target zero set:

```text
1,3,5,7,11,13,15,17,19,21,23,25,27,29,31,33,35,37,44,46,48,49,50,52,54,56,58,60,62,64
```

Right raw/current/future zero set (identical in all three domains):

```text
0,2,4,6,8,10,14,16,18,22,24,26,28,32,36,40,41,43,45,47,49,51,53,55,57,59,61,63,65
```

The left-hand 28-versus-30 difference is fully explained by exact-stride
sampling: episode 15 is active only at raw indices 418/419 (modulo five 3/4),
and episode 48 only at 512/513/514 (modulo five 2/3/4). Neither contains a raw
active sample at an index divisible by five.

The processed proxy is often held when its tracker is inactive: 7,331 of
11,005 left closed-proxy frames and 7,842 of 10,621 right closed-proxy frames.
This is why the proxy cannot be trained unmasked. There are 110 left and 118
right proxy transitions; only 54/58 respectively have tracker validity at both
adjacent frames.

## Production contract and verification

- The dataset sidecar supplies current `[T,2]`, future `[T,48,2]`, and
  record-only adjacent-pair masks. Edge padding is false; it is never
  last-value held.
- Production derives the action-time mask from equal bool
  `action_delta_eef_is_pad`/`hand_status_is_pad` source vectors, slices source
  slot `f`, and requires the remaining `f+1..f+48` horizon to be exactly H=48.
  This zeros wrist and grip loss together at clamped tail positions.
- Per-frame action MSE uses grip weights
  `left_active[target]*0.25` and `right_active[target]*0.25`. Wrist dimensions
  0..17 are unchanged, time padding zeros every dimension, and model-padding
  dimensions 20..31 remain zero-weight.
- The transition-pair mask is emitted for a future explicit ablation but no
  transition auxiliary exists or consumes it.
- Norm v3 excludes inactive proxy values from state/action grip statistics.
  Valid sample counts for dimensions 18/19 are 7,087/6,574 for state and
  335,074/308,116 for action. Other source dimensions use all 20,013 states and
  883,008 valid H=48 action targets; padding dimensions use zero samples.
- Synthetic H=48 checks covered `f+1`, tail padding, target-1/target pair
  semantics, left/right dimensions 18/19, time-mask composition, and the exact
  legacy no-sidecar behavior. A `labvla` CPU-only (`CUDA_VISIBLE_DEVICES=''`)
  transform/trainer run confirmed inactive grip errors contribute zero loss,
  active grip uses 0.25, wrist errors remain supervised, and record-only pair
  masks are not trainer inputs. Strict dataset SHA/load and real bundle/norm
  preflight replay also passed.

## Bound artifacts

The sidecar meta, norm v3 and report carry the same WAM code-manifest digest.
The manifest includes dataset fingerprinting, verified I/O, the tracked Giga
LeRobot loader and every grip generation/runtime path. The source receipt also
binds the installed LeRobot dataset implementation bytes/version as a separate
runtime dependency; its absolute import path is record-only.

```text
dataset_id: ego-processed-sha256:b5308e8a6882e128f324e097c41e817ad1a5230609aa0e20edce8caba2aebd5d
dataset_source_receipt_sha256: d4873e6c0670a07aed9d48e4fea064ce56b402a408fff241bf6c3d243278d552
raw_hdf5_bundle_sha256: f074948ce59928b90f3122b205693590e74a114c1f54172ff41e938fe84db6cb
final_hdf5_bundle_sha256: 334a8aeb66d90853bbbba2af044153b624018d008b45515fe0840dc4858448ed
hand_active_sidecar_bundle_sha256: 72fff6f1a17916cc76cb0057eb77350e56cd1b883bd69759fe93131efc023fe9
hand_active_meta_sha256: ed9eea54a36a9d1971b9a5d96174510cf4b8d8c423a648af5fe095eb8d81e090
norm_v3_sha256: 4f1bcfc4df9b0c088b1476722acb7c4cacfefe9597e31e443762a7b6f47bde8a
wam_code_receipt_sha256: 0351922c6724cf9e2c64b42175a49d2c08eb29532bb4750c40932e117e962a54
lerobot_runtime_source_sha256: bb4b1f7cb2dca6b440f846b35670eba2ddb39434db0f03591a53fef6f022a9f8
source_receipt_sha256: 34898876b4a843f52547c6e7d1a97f62771c82cadc659fc06c4647b0ab09835a
reviewed_report_sha256: 75b3f692ad179847f300d62c24549c32423b56ff6b68e611a64fe564e627faae
```
