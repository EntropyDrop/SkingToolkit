python ./infer.py \
  --checkpoint ./runs/foreground_alpha_test2/best.pt \
  --inputs ../test_imgs/banana_output1.png \
  --output_dir ../test_imgs
  ##--bg_color 0,0,0 \
  ##--uncompose