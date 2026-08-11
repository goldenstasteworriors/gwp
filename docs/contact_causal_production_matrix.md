# Contact causal-main production matrix v1

op42 adds a separate, versioned admission chain for Robot Stage2. It does not
extend or reinterpret the historical `soft_semantic / hard_st / hard_detached`
mechanism matrix. The new schema and ID are:

- `contact-causal-production-matrix/v1`;
- `contact-asym25-causal-main-q12-b16-stage2/v1`;
- exact routes `aux_only_direct_q12`,
  `equal_bit_unstructured_hard_st`, and `hard_st_semantic`.

The checked-in matrix is
`configs/contact_causal_production_matrix.v1.json`. Its scientific-admission
flags are false. The checked-in v7/v3/v2/v5/v3 templates contain only
false/null authorization evidence and intentionally fail their validators.
They are not examples of valid evidence and cannot authorize training.

## Causal contrast

All routes use Q=12 planner queries and the same 22D semantic auxiliary
decoder. `aux_only_direct_q12` keeps the planner and auxiliary forward matched,
but action attends the reference directly. It explicitly does not claim equal
action-visible capacity, backward FLOPs, or exact compute. The two strict
routes use the same 22-to-action encoder and have exactly 48,673 valid states
per query and 186.85000894308362 bits across Q12. The equal-bit route exposes
an unstructured code; the semantic route exposes the finite contact grammar.
The equal-bit mixing matrix is nonpersistent and cannot enter a checkpoint
state dict.

The route-local and seed-derived differences are the exact JSON pointers in
the matrix manifest. The canonical comparator replaces only those leaves and
then compares the complete receipt. A new or unknown field is stable by
default. Direct's inactive parameters must be exactly
`contact_condition_encoder.*`; they may have no gradient, optimizer state, or
drift. The strict routes may not declare unused parameters.

## Route-neutral Ego precursor

The new precursor ID is `ego-contact-planner-precursor-q12/v1`. Its independent
entry is
`configs/giga_world_policy_0_5_ego_contact_causal_precursor.py`. The v7 gate
must bind human-adjudicated contact labels, fixed save step, current HEAD,
source tree, data, normalization, reviewer identity, expiry, and a unique
training nonce. The trainer freezes the complete action path before optimizer
construction. The scientific forward accepts visual/state/planner inputs and
returns visual and contact predictions, with no action input, action target,
action loss, or action output. `hand_active` and grip are never contact ground
truth.

Its canonical precursor receipt binds frozen/trainable parameter sets, tensor
digests, objective, checkpoint bundle, and external review. All nine Stage2
runs must use byte-identical receipt bytes, the same bundle, and the same
step-0 parameter tensor digest.

## Robot v3 admission chain

The Robot entry is
`configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_causal_matrix_finetune.py`.
It bootstraps the v3 gate before importing a generic asymmetric25 base config;
it never imports the old contact-production config. It requires exact,
explicit paths for the matrix, precursor receipt, dataset-level contact
coverage receipt, training-data v2 receipt, preflight v2 receipt, collector v3
config, and reviewed checkpoint-receipt v5 base. The selected route, seed,
training run ID/nonce, external acceptance, current HEAD/source tree,
canonical25 target-hardware data, robot-reviewed labels, train-only
normalization, and precursor tensor binding must agree across those files.

The required route selectors are:

```bash
export GWP_CONTACT_CAUSAL_MATRIX_ID="contact-asym25-causal-main-q12-b16-stage2/v1"
export GWP_CONTACT_CAUSAL_ROUTE="hard_st_semantic"
export GWP_MATCHED_CONTROL_VARIANT_ID="$GWP_CONTACT_CAUSAL_ROUTE"
```

换行位置：以上每条 `export` 独占一行。route 必须逐字替换为三个 exact route 之一，不能写
`hard_st`、`semantic` 或其它 alias。

Additional required evidence paths are named
`GWP_CONTACT_CAUSAL_TRAINING_GATE`,
`GWP_CONTACT_CAUSAL_MATRIX_MANIFEST`,
`GWP_EGO_CAUSAL_PRECURSOR_RECEIPT`,
`GWP_CONTACT_CAUSAL_DATASET_COVERAGE_RECEIPT`,
`GWP_CONTACT_CAUSAL_TRAINING_DATA_RECEIPT`,
`GWP_CONTACT_CAUSAL_PREFLIGHT_RECEIPT`,
`GWP_CONTACT_CAUSAL_COLLECTOR_CONFIG`, and
`GWP_CONTACT_CAUSAL_CHECKPOINT_RECEIPT_BASE`. This repository intentionally
contains no true values for them.

Dataset-level supervision must include at least one reviewed positive contact
event and nonzero grouped coverage. A particular training window may have zero
supervision; in that case supervision v2 produces canonical positive zero with
zero planner gradient and does not rewrite unknown contact to a negative.
Model-side oracle input is rejected for all three predicted routes. Oracle is
only a separately reported upper bound and cannot enter the main matrix.

## Checkpoints and evaluation

Each of 3 seeds × 3 routes is one unique training run. Every run emits steps
0, 100, and 200, producing exactly 27 independently accepted checkpoints.
Step 0 precedes optimizer updates. Parent binding is
precursor/base → step 0 → step 100 → step 200. The v5 receipt records parameter
and buffer tensor digests, config and whole-bundle digests, topology,
optimizer/scheduler state, labels, windows, noise/RNG, precursor, route,
trace, gate, preflight, dataset, norm, hardware, current HEAD/source tree, and
expiry-bound external identities. Step 200 is the preregistered primary;
held-out best-step selection is forbidden.

The route-aware evaluator is v2, result is v3, aggregate is v5, and checkpoint
manifest is v3. All three routes obtain contact field metrics from the same
semantic auxiliary logits. Direct reports `none_direct_reference` and no fake
packet. Equal-bit reports only codec validity/utilization/entropy/invalid-zero
metrics. Semantic reports grammar/field metrics. Action targets, masks, and
denormalization are shared.

## CPU implementation acceptance

The following command runs only synthetic CPU tensors and creates no retained
checkpoint evidence:

```bash
CUDA_VISIBLE_DEVICES='' conda run -n dit4dit python -B scripts/audit_contact_causal_production.py
```

换行位置：该命令为单行。The audit builds 27 real, nonempty, loadable tiny
weight files under the project output mount, validates the complete matrix and
negative cases, and removes the temporary files before returning. It also
compares 15 protected legacy files byte-for-byte with the op42 baseline. A
passing synthetic audit is implementation evidence only: it does not claim
benefit, admit production data, or authorize server/5B training.

The v3 trace comparator accepts exactly one trace file per new route argument:

```bash
CUDA_VISIBLE_DEVICES='' conda run -n dit4dit python -B scripts/compare_contact_causal_matrix_hashes.py \
  --seed 6666 \
  --aux-only-direct-q12 /ABS/PROJECT/EVIDENCE/direct.jsonl \
  --equal-bit-unstructured-hard-st /ABS/PROJECT/EVIDENCE/equal.jsonl \
  --hard-st-semantic /ABS/PROJECT/EVIDENCE/semantic.jsonl
```

换行位置：每个行尾反斜杠后换行；替换 seed 和三个项目盘绝对路径。不要使用 `/tmp`。

## Current decision

The implementation remains No-Go for scientific production. Real target
hardware data, human review, coverage/agreement, oracle sufficiency, matched
traces, precursor training, external reviewer acceptance, and 27 real
checkpoint acceptances do not exist in the repository. op42 must not create or
self-sign them.
