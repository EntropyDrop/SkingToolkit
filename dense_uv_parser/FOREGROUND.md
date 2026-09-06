# Minecraft foreground model

This model adapts the locally cached BiRefNet decoder to renderer-supervised Minecraft alpha. The pretrained backbone is frozen and checksum-verified after training. This is a separate model from the v61/v101 UV parser.

## Data and validation

`prepare_foreground_data.py` renders 4,096 training, 128 validation and 128 test skin identities in two views. Normalized RGBA hashes prohibit identical textures across splits. Premultiplied affine resampling avoids hidden transparent RGB leaking into targets. Seven background families include near-foreground colours, gradients, noise, and checkerboards; JPEG and border-touching crops are included. Real regression images never enter gradient training.

`train_foreground.py` saves the exact source files, dataset and pretrained weight hashes, decoder-only checkpoints and validation history. Admission requires IoU >= 0.985, boundary F1 no more than 0.003 below baseline, at least 2% lower boundary alpha error, and >= 99.5% recall on annotated hat/glasses foreground. Model selection uses validation only; selected weights are evaluated on held-out test identities against the pretrained baseline.

The first 512-pixel pilot was rejected for false-positive background islands and boundary F1 regression. The second run uses the same 1024-pixel resolution as deployment and hard-background pixel loss. Final measurements and the accepted checkpoint are recorded in the release report.

## Inference

`foreground_batch.py` runs the foreground model in the existing `comfy` environment, then passes the exact ordered probability paths to the parser in `sking-v61-worker`. It requires explicit foreground and parser checkpoints. Input bytes, model deltas, settings and outputs are fingerprinted. No packages are installed into either production environment.

The silhouette uses alpha >= 0.5. UV RGB samples additionally require alpha >= 0.98 and a one-pixel interior margin at parser resolution. This source mask does not erase thin foreground geometry. It avoids sampling uncertain mixed boundary RGB; it does not perform full foreground colour estimation or promise exact physical alpha on generated images.

The existing UV parser checkpoint remains unchanged until the subsequent brim-data training has passed its own checks.
