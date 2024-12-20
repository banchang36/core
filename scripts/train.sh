#!/bin/bash
WANDB_KEY=$WANDB_KEY # if wandb is enabled, set WANDB_KEY to log in
base_dir="."
data_dir="${base_dir}/gec_private_train_data"
detect_vocab_path="./data/vocabulary/d_tags.txt"
correct_vocab_path="./data/vocabulary/labels_vi.txt"
train_path="${data_dir}/train-stage2.edits"
valid_path="${data_dir}/test.edits"
config_path="configs/ds_config_basic.json"
timestamp=`date "+%Y%0m%0d_%T"`
save_dir="./ckpts/"
pretrained_transformer_path="vinai/phobert-base"
mkdir -p $save_dir
cp $0 $save_dir
cp $config_path $save_dir

# Clear CUDA cache before starting
python train.py --wandb --wandb_key --config_path configs/ds_config_basic.json --num_epochs 50 --max_len 256 --train_batch_size 8 --accumulation_size 1 --valid_batch_size 8 --cold_step_count 0 --lr 1e-5 --cold_lr 1e-3 --skip_correct 0 --skip_complex 0 --sub_token_mode average --special_tokens_fix 1 --unk2keep 0 --tp_prob 1 --tn_prob 1 --detect_vocab_path ./data/vocabulary/d_tags.txt --correct_vocab_path ./data/vocabulary/labels_vi.txt --do_eval --train_path ./gec_private_train_data/train-stage2.edits --valid_path ./gec_private_train_data/test.edits --use_cache 1 --save_dir ./ckpts/ --pretrained_transformer_path vinai/phobert-base 2>&1 | tee ./ckpts/train-20241104_19:10:00.log
echo ${run_cmd}
eval ${run_cmd}
