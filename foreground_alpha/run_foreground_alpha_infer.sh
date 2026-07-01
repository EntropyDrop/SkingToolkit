python ./infer.py \
  --checkpoint ./runs/foreground_alpha_test1/best.pt \
  --inputs ../test_imgs/banana_output1_front.png ../test_imgs/banana_output1_back.png \
  --output_dir ../test_imgs
  ##--bg_color 0,0,0 \
  ##--uncompose