# v101 baseline before v102 (2026-09-06)

The current v101 release is `crown_geometry_20260906`, committed at `3914388`.
Checkpoint: `runs/v101_crown_geometry_release_20260906/parser.pt`.
SHA256: `a8aa3d8cd51cc6fec28d7c525aa1b012976d7cb1206e8b00c707de76bfb205dc`.
All 1,636 dataset results were regenerated from `*_edited.png`; source inputs are obsolete.

Three new user-reported examples are recorded in `regression/v102_development_cases.json`.
They are development regressions, not independent test data and not gradient inputs.
Raw baseline logits, routing, UVs and layers are in `runs/v102_audit_20260906`.

Confirmed findings:
- Blue crown 1WQEQRUDRLMLVX3E: royal presence = [0.0026441, 0.9680233]. The two-view minimum gate disables all royal routing. Top cap material consequently remains in the inner layer. The old crown data only authors a one-cell perimeter, with no inward extensions.
- Nose 1ZDZB1UAMS1DEZLD: at outer UV (43,13), ownership predicts inner face with mean 0.997, but the later headwear branch predicts crown with mean 0.852 and promotes the facial region again. This is a disagreement between semantic heads, not merely a missing face veto.
- Beard 1ZW63H5Z8YNV57MT: facial hair is split between inner and outer layers. Outer top cell (45,6) routes 43 pixels with mean inner-hair probability near 1.0. The ownership correction currently consumes face evidence only, not inner-hair evidence. Existing synthetic faces have eyes, brows and mouths but no authored noses or beards.

v102 should learn from varied synthetic geometry, materials, facial features and paired-view context; it must not use filename/colour rules or force all hair to inner/all faces symmetric. Keep prior crown openings/caps, glasses forehead, headphone colour isolation and continuous hat-band/brim regressions. Prior test source identities have been used repeatedly; report subsequent measurements as regressions, not fresh held-out generalization.

No released checkpoint, default registry or dataset result is changed by this baseline commit.
