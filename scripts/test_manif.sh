python test.py \
  --mode kpcn \
  --use_llpm_buf True \
  --input_channels 39 \
  --diffuse_model trained_model/kpcn_manif_2/diff_e6.pt \
  --specular_model trained_model/kpcn_manif_2/spec_e6.pt \
  --path_diffuse_model trained_model/kpcn_manif_2/path_diff_e6.pt \
  --path_diffuse_model trained_model/kpcn_manif_2/path_spec_e6.pt \
  --data_dir '/root/kpcn_data/kpcn_data/data' \
  --save_dir 'test/kpcn_manif_2'