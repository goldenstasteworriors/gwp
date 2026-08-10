# Finite-capacity contact semantic transport

Idea card: `762fb9fc-616f-4d2a-8164-f68ba95da21c`

## Why f3ae3bf is not a strict semantic channel

The v1.1 action stream reads a 22-D field-structured tensor instead of the
planner's 1024-D hidden state, but the tensor is still continuous. Binary and
categorical probabilities can encode information in their fractional values,
and `start_tau`/`end_tau` are real values in `[0,1]`. Therefore its theoretical
capacity is not bounded by the semantic field cardinalities. This is a
low-dimensional routing pilot, not yet a finite-capacity bottleneck.

## Transport contract

- `soft`: identity. It preserves the f3ae3bf tensor and gradient behavior for
  the compatibility control.
- `hard_st`: forward values are finite and grammar-valid. Backward uses the
  straight-through estimator so action loss can update the planner.
- `hard_detached`: the same finite forward values, detached before the
  22-to-action-width encoder. Only explicit contact supervision updates the
  planner.
- Presence and censor fields use deterministic hard Bernoulli decisions.
- Side, site, object role, and mode use deterministic hard one-hot argmax.
- Start and end time use `B` uniformly spaced levels including 0 and 1.
- Uncensored segments satisfy `start_bin <= end_bin`. Left-censored segments
  fix `start_bin=0`; right-censored segments fix `end_bin=B-1`.
- An absent segment has one canonical representation: presence zero and all
  remaining fields zero. This prevents unused fields in padded queries from
  becoming a side channel.
- The action expert receives only the transported 22-D tensor through the
  existing 22-to-action-width encoder. It never receives planner hidden state.
- All modes reject NaN and positive/negative infinity before transport.
  `hard_st` uses `hard.detach() + (semantic - semantic.detach())`: the
  parenthesized term is exactly zero in the forward pass, so its bytes equal
  `hard_detached` instead of retaining float-cancellation residues from the
  continuous semantic value. The derivative with respect to `semantic`
  remains the identity straight-through surrogate.

Masked oracle values are merged at field level before hard transport. Known
fields are therefore validated by the same finite grammar as predictions;
unknown fields use predictions. The `soft` oracle path retains the f3ae3bf
stop-gradient behavior exactly. `hard_st` allows action gradients through
unknown predicted fields, while `hard_detached` blocks all action-to-planner
gradients. Cached action-only inference stores the transported tensor, so later
denoising calls cannot recover continuous planner values.

## Strict capacity bound

There are four valid timing/censor cases for an active segment:

```text
uncensored: B(B+1)/2
left-censored only: B
right-censored only: B
both censored: 1
```

Including the single canonical absent state gives

```text
N_time,censor = B(B+1)/2 + 2B + 1
N_segment = 1 + (3*6*4*4) * N_time,censor
bits_segment <= log2(N_segment)
bits_Q <= Q * log2(N_segment)
```

For `Q=12`:

| Time bins | Valid time/censor states | Strict states/segment | Strict bits/segment | Strict bits for Q=12 |
|---:|---:|---:|---:|---:|
| 8 | 53 | 15,265 | 13.897940 | 166.775280 |
| 16 | 169 | 48,673 | 15.570834 | 186.850009 |
| 32 | 593 | 170,785 | 17.381822 | 208.581861 |

The metadata additionally records the pre-canonicalization loose upper bound
`1 + 1152*B^2` so invalid combinations cannot silently reappear without a
reported capacity change. These are representation-alphabet bounds, not
empirical mutual-information estimates. The model logs the selected mode,
bins, query count, strict state count, and bit bound at construction; the
opt-in config also stores them under `contact_transport_metadata`. `soft`
explicitly reports that it has no auditable finite semantic alphabet bound;
this does not claim infinite precision for a machine float.

## Literature evidence affecting the architecture

- VQ-VAE uses a discrete forward code and copies decoder gradients to the
  encoder with a straight-through estimator. This supports `hard_st`, but also
  warns that forward discreteness does not make the gradient unbiased:
  <https://arxiv.org/abs/1711.00937>.
- Bengio et al. define the straight-through estimator as copying the gradient
  through a hard/stochastic unit. This is the direct optimization precedent:
  <https://arxiv.org/abs/1308.3432>.
- FSQ obtains a finite implicit codebook as a product of fixed scalar levels,
  without a learned VQ codebook. This motivates fixed time bins and an exact
  capacity calculation instead of adding an unnecessary learned codebook:
  <https://arxiv.org/abs/2309.15505>.
- DreamerV2 shows that categorical discrete latents can support a world model,
  but its latent categories are not guaranteed to be human-interpretable:
  <https://arxiv.org/abs/2010.02193>.
- VQ-BeT applies hierarchical vector quantization to latent robot actions and
  improves multimodal behavior generation. It is a discrete-action precedent,
  not a contact-semantic or WAM bottleneck precedent:
  <https://arxiv.org/abs/2403.03181>.
- Contact-conditioned locomotion demonstrates that future contact location and
  timing can condition a general low-level policy and improve OOD behavior. It
  is the closest contact-plan-to-policy precedent, but it uses expert contact
  goals for locomotion rather than video-predicted manipulation contact tokens:
  <https://arxiv.org/abs/2408.00776>.
- GeCCo extends contact-plan conditioning to general loco-manipulation with
  arbitrary contact points and durations. It strengthens the case for a
  planner/controller contact interface, but its plans are supplied or sampled
  geometric goals rather than a finite semantic alphabet predicted inside a
  video world-action model: <https://arxiv.org/abs/2509.17582>.
- Contact-Anchored Policies replace language with spatial contact anchors for
  reusable utility policies. This is a direct collision with broad claims that
  contact conditioning itself is novel; it does not provide transition-token
  dynamics, an exact channel bound, or the WAM causal-routing tests proposed
  here: <https://arxiv.org/abs/2602.09017>.
- Mask World Model predicts semantic masks as a geometric world-model
  bottleneck. It is the closest WAM-level semantic-factorization precedent,
  but masks and their learned features do not define an auditable finite
  contact grammar or isolate contact-plan-to-action causality:
  <https://arxiv.org/abs/2604.19683>.
- DexMani transfers human contact-conditioned manipulability evolution across
  dexterous hands. It strongly supports the cross-embodiment contact-transition
  hypothesis, while using an energy prior rather than an action-visible finite
  token bottleneck: <https://arxiv.org/abs/2608.00554>.
- The local Zotero note on FAST documents that quantization capacity and
  compression fidelity materially affect high-frequency robot policies. FAST
  tokenizes actions, not a causal semantic plan, so it is a rate-distortion
  control rather than a novelty collision:
  `PaperNotes/多模态/FAST.md` (Zotero item `CBLGABU2`).

The defensible novelty claim is therefore not “discrete latents” or “contact
conditioning.” It is the combination of an explicit contact-transition
grammar, a strict action-visible information bound, video-side prediction, and
paired causal controls inside an action-centered WAM. This search found no
paper combining all four; that is a retrieval result, not proof of absence.

## Controlled causal ablation

The machine-readable matrix is
`configs/contact_transport_ablation_matrix.json`. Every variant uses seeds
`6666/7777/8888`. A paired comparison is invalid unless batch size, ordered
sample IDs, raw batch, sampled augmentation parameters, visual/action
timesteps, and visual/action noise hashes match.

This matrix is an experiment contract, not evidence that all routes exist.
Each variant records `implementation_status`, exact config overrides, and the
action-visible route. Only `soft`, `hard_st`, and `hard_detached` transport are
implemented by this change; the existing trainer/model also contain the
masked-oracle merge, but audited labels are still missing. The Q=12 direct
capacity control, auxiliary-only route, raw-planner-hidden route, shuffle, and
delay controls remain explicitly `spec_only` or blocked. In particular, Q=12
direct means planner computation is retained only as a parameter/FLOP control
and every planner output is discarded before the direct action stream; the
current config cannot express that route.

The rank-local hash collector and offline comparator are implemented for the
exact runner set `soft_semantic`, `hard_st`, and `hard_detached`. Fixed seeds
alone still do not establish a matched comparison, so no result may be called
paired until the external gate binds non-empty per-step artifacts and all six
required hashes are equal.

The required variants separate token-count capacity, auxiliary supervision,
raw hidden routing, soft semantics, finite semantics, gradient route, oracle
upper bound, and negative controls (`shuffled`, `delayed`). Shuffling occurs
within a batch after masks are applied. Delay shifts by exactly one action
chunk and masks the leading invalid condition instead of wrapping around.

Do not start 5B or server training until both the v1.1 label coverage audit and
the asymmetric-25D converter/runtime gate pass. Tiny CPU tests and static
checks are necessary but not sufficient to open either gate.
