python ./infer.py \
  --checkpoint ./runs/foreground_alpha_test2/best.pt \
  --inputs ../../SkingDataset/DDJ_real2render/test_output/img2_template7.png \
  --output_front ../test_imgs/front_rgba.png \
  --output_back ../test_imgs/back_rgba.png
  ##--output_dir ../test_imgs \
  ##--bg_color 0,0,0 \
  ##--uncompose