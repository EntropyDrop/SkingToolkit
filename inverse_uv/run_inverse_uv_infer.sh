python3 ./infer.py \
  --checkpoint ./runs/inverse_uv_test17/best.pt \
  --output ../test_imgs/output.png \
  --mappings_dir ../../github/differentiable_minecraft_renderer/mappings \
  --view_images ../test_imgs/banana_front_rgba.png ../test_imgs/banana_back_rgba.png 
  #--view_images ../test_imgs/banana_output1_front_rgba.png ../test_imgs/banana_output1_back_rgba.png 
  #--view_images ../test_imgs/front.png ../test_imgs/back.png 