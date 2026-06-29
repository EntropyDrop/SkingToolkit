python3 inverse_uv/infer.py \
  --checkpoint runs/inverse_uv_test12/best.pt \
  --output pred_uv.png \
  --mappings_dir ../github/differentiable_minecraft_renderer/mappings \
  --view_images alpha_test/banana_front_rgba.png alpha_test/banana_back_rgba.png 
#--view_images inverse_uv/test_input/walk_perspective_ortho_pyvista.png inverse_uv/test_input/walk_perspective_back_ortho_pyvista.png 