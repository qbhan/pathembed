python train_decomp_origin.py \
  --mode 'decomp_kpcn' \
  --device 'cuda:1' \
  --input_channels 34 \
  --hidden_channels 100 \
  --num_layer 9 \
  --eps 0.00316 \
  --do_val \
  --lr 1e-4 \
  --epochs 40 \
  --loss 'L1' \
  --data_dir '/mnt/ssd2/kbhan/KPCN' \
   --use_llpm_buf True
