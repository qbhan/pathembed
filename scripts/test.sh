python test.py \
  --mode kpcn \
  --num_layers 9 \
  --kernel_size 5 \
  --device 'cuda:1' \
  --diffuse_model trained_model/kpcn/diff_e12.pt \
  --specular_model trained_model/kpcn/spec_e12.pt \
  --data_dir '/mnt/ssd2/kbhan/KPCN' \
  --save_dir 'test/kpcn'
