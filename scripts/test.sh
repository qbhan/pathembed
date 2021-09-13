python test.py \
  --mode kpcn \
  --diffuse_model trained_model/simple_kpcn_1_finetune_1/diff_e8.pt \
  --specular_model trained_model/simple_kpcn_1_finetune_1/spec_e8.pt \
  --data_dir '/root/kpcn_data/kpcn_data/data' \
  --save_dir 'test/test_simple_kpcn_1_finetune_1'
