python train.py \
  --data_dir ../skins \
  --output_dir runs/inverse_uv_test14 \
  --views walk_front_both_layer_ortho,walk_back_both_layer_ortho \
  --batch_size 16 \
  --epochs 100 \
  --val_split 0.1 \
  --mappings_dir ../../github/differentiable_minecraft_renderer/mappings \
  --save_every 1 \
  --preview_every 1