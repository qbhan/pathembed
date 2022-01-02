python test_valid.py \
  --mode kpcn \
  --device 'cuda:0' \
  --use_llpm_buf True \
  --input_channels 39 \
  --diffuse_model trained_model/kpcn_manif_valid_both/diff_e12.pt \
  --specular_model trained_model/kpcn_manif_valid_both/spec_e12.pt \
  --path_diffuse_model trained_model/kpcn_manif_valid_both/path_diff_e12.pt \
  --path_diffuse_model trained_model/kpcn_manif_valid_both/path_spec_e12.pt \
  --data_dir '/mnt/ssd2/kbhan/KPCN' \
  --save_dir 'test/kpcn_valid_both'