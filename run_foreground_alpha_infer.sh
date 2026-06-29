python foreground_alpha/infer.py \
  --checkpoint runs/foreground_alpha_test1/best.pt \
  --inputs inverse_uv/test_input/banana_front.png inverse_uv/test_input/banana_back.png \
  --output_dir alpha_test
  ##--bg_color 0,0,0 \
  ##--uncompose