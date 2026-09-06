# v101 retraining after foreground acceptance

The foreground release in `foreground_release.json` passed validation, 128 held-out skin identities / 256 renders, and 14 real development cases before brim-data work began. Original v100/v101 commits and weights remain unchanged.

The new hat generator authors inner crown, inner band, outer crown, outer band and brim components separately from accessory occupancy. It includes full wraparound rings (75%), front-only visors (15%) and partial/asymmetric brims (10%). Stepped hats occur in 70% of hats. Materials and band colours vary independently of geometry, and both views share the authored UV object. No real development image enters gradient training.

A six-class component head supplements the existing four-class accessory head. Confident inner crown/band predictions can correct an older outer route, while original v61 parameters remain frozen. Cross-face UV losses and procedural foreground supervision remain active. Unknown raw inner-layer pixels are not used as semantic negatives (`--replay-weight 0`).

The 1,600-step pilot achieved about 98% brim recall in both real views but was rejected for colour-band loss and glasses precision. The colour failure exposed a routing/sampling mismatch: newly opaque outer texels cover pixels previously attributed to the inner surface, but their colours were sampled only from the promoted subset. `material_refine.py` now fits visible head RGB against both canonical source views, retaining the predicted alpha and all body texels exactly. It accepts only a lower reconstruction error and uses the same reliable foreground colour sources as UV sampling. This is inference-time material fitting, not training on development images.

Final checkpoint admission requires synthetic precision/recall and completeness, the existing stored-layer disagreement guard, glasses precision >= 0.85 and whole-glasses outer recall >= 0.90, both real brim recalls >= 0.90, and both annotated colour-band chroma retention values >= 0.95. The colour metric is specific to this red-band development case; it is not a universal semantic metric.

Synthetic validation uses renderer alpha as foreground, with the same reliable-source filtering and complete rendering pipeline; real validation uses cached predictions from the accepted foreground model. Model selection uses validation, followed by held-out evaluation and full real-image visual review.
