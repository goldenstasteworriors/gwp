# Contact-Transition WAM pilot

This pilot tests one falsifiable question before building a large weak-label
pipeline: does a current-observation contact plan improve action generation when
the action stream cannot directly read the reference-image tokens?

## Causal routing

The action expert uses `[state, contact queries, noisy action]`; the visual
expert uses `[reference, future]`.

```text
state/reference <- state/reference
contact         <- state/reference/contact
action          <- state/contact/action        # strict pilot
future          <- all
```

`contact_bottleneck=False` restores a direct reference-to-action edge for the
matched-capacity auxiliary-head baseline.  With zero contact queries the model
is backward compatible with the original GWP-0.5 topology.

## Ordered event target v1

Events are sorted by `(onset time, side, canonical site)` and padded to `Q`.
The 54-D dense target is a concatenation of:

| Field | Width | Values |
|---|---:|---|
| presence | 1 | binary |
| side/edge type | 3 | left, right, object-object |
| canonical site | 7 | opposer, primary, aux1-3, palm, tool |
| object role | 4 | target, tool, receptacle, fixture |
| tracked part slot | 8 | clip-local slots |
| contact mode | 6 | free, stick, slide, roll, impact, unknown |
| transition | 5 | none, onset, hold, switch-to-slide, release |
| object effect | 7 | none, translate, rotate, articulate, flow, deform, transfer |
| event time bin | 4 | event offset within the 48-frame action horizon |
| duration bin | 4 | short to long |
| part-local 2D region | 2 | normalized coordinates |
| visible normal proxy | 3 | camera-frame unit vector |

Categorical fields are one-hot; the final five values are continuous.  The MVP
uses elementwise BCE (or Smooth-L1 via config), plus `contact_loss_mask` and
`contact_confidence`.  A grouped categorical loss and Hungarian matching are
deliberately deferred until the ordered-label oracle gate is passed.

Batch keys:

```text
contact_target          [B, Q, 54]
contact_loss_mask       [B, Q, 54] or broadcastable
contact_confidence      [B, Q, 54] or broadcastable
contact_condition       [B, Q, 54] optional
contact_condition_mask  [B, Q, 1] optional
```

`scripts/build_contact_transition_sidecar.py` converts interval annotations
from their source video FPS to the processed LeRobot FPS and emits one NPZ per
episode.  Annotated episodes supervise padded-query absence plus known event
fields.  Unannotated episodes have an all-zero loss mask, so missing annotation
is never treated as negative contact.  A manifest can leave uncertain groups
(finger site, object part, effect, local region, or normal) unspecified; their
field-level masks stay zero.  Set `GWP_CONTACT_SIDECAR_DIR` to load these files
through `WAMLeRobotDataset`.

`contact_condition_mode=predicted` never injects future labels.  `oracle` uses
`contact_target` only for the oracle upper-bound experiment.  The oracle model
must not be reported as a deployable policy.

## EgoHumanoid action adapter

The processed EgoHumanoid LeRobot export has no `observation.state`.  It stores
an absolute dual-wrist pose as `[xyz, qx, qy, qz, qw] x 2`, local wrist motion
as `[xyz, roll, pitch, yaw] x 2`, and a two-value hand-closure proxy.  The ego
adapter converts both wrists to `position(3) + rotation-6D(6)` and appends the
left/right closure proxy, yielding 20 valid dimensions before padding to 32.

The closure proxy was produced by fitting a square wave to tracked hand motion;
it is not a per-finger contact label and must not be presented as dexterity
ground truth.  Dexterity supervision remains in the contact-transition sidecar
and, in the next data version, palm-local functional-site trajectories.

The source defines `action_delta_eef[i]` as the motion from pose `i-1` to `i`.
`giga_world_policy_0_5_ego_contact_pretrain.py` therefore loads offsets `0..48`,
uses offset zero only for the current grip state, and predicts offsets `1..48`.
`compute_ego_wam_norm.py` uses the identical clamped future-window semantics.
Padded action dimensions are emitted through `action_dim_mask` and contribute
neither flow noise nor action loss.

## Stage gate

Continue to mesh-free weak labelling only if:

1. oracle contact improves part-sensitive success by at least 10 points and
   correct-first-contact by at least 15 points over GWP-0.5;
2. predicted contact recovers at least 70% of the oracle gain;
3. strict bottleneck beats the direct-visual auxiliary-head baseline by at
   least 5 points on three OOD splits;
4. shuffled/delayed contact produces the intended phase-specific action change;
5. cached action-only latency increases by no more than 10%.

The first hardware schema should use 25 physical dimensions before padding to
32: left/right wrist `9+9`, left RH56DFTP `6`, right OmniPicker aperture `1`.
The OmniPicker still exposes two canonical jaw contact sites even though it has
one active actuator.
