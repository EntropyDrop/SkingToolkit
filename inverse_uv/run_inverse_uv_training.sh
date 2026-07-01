
# 1. 限制 PyTorch 的 CPU 并行线程，防止 256 线程严重锁竞争
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python train.py \
  --data_dir ../skins \
  --max_samples 10000 \
  --output_dir runs/inverse_uv_test17 \
  --views walk_front_both_layer_ortho,walk_back_both_layer_ortho \
  --batch_size 16 \
  --num_workers 16 \
  --epochs 100 \
  --val_split 0.1 \
  --mappings_dir ../../github/differentiable_minecraft_renderer/mappings \
  --save_every 1 \
  --preview_every 1 \
  --augment \
  --perspective_scale 0.01 \
  --distortion_scale 0.005 \
  --translation_scale 0.02