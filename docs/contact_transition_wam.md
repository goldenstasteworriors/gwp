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

## Ordered event target v0

Events are sorted by `(onset time, side, canonical site)` and padded to `Q`.
The initial 50-D dense target is a concatenation of:

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
| duration bin | 4 | short to long |
| part-local 2D region | 2 | normalized coordinates |
| visible normal proxy | 3 | camera-frame unit vector |

Categorical fields are one-hot; the final five values are continuous.  The MVP
uses elementwise BCE (or Smooth-L1 via config), plus `contact_loss_mask` and
`contact_confidence`.  A grouped categorical loss and Hungarian matching are
deliberately deferred until the ordered-label oracle gate is passed.

Batch keys:

```text
contact_target          [B, Q, 50]
contact_loss_mask       [B, Q, 50] or broadcastable
contact_confidence      [B, Q, 50] or broadcastable
contact_condition       [B, Q, 50] optional
contact_condition_mask  [B, Q, 1] optional
```

`contact_condition_mode=predicted` never injects future labels.  `oracle` uses
`contact_target` only for the oracle upper-bound experiment.  The oracle model
must not be reported as a deployable policy.

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
