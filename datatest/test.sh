#!/bin/bash

SUBSET="testdata"
SOURCE="./datatest/source_test.txt"
TARGET="./datatest/target_test.txt"
OUTPUT="./gec_private_train_data/${SUBSET}.edits"
python utils/preprocess_data.py -s $SOURCE -t $TARGET -o $OUTPUT

base_dir="."
python predict.py \
    --batch_size 32 \
    --iteration_count 5 \
    --min_len 3 \
    --max_len 128 \
    --min_error_probability 0.0 \
    --additional_confidence 0.0 \
    --sub_token_mode "average" \
    --max_pieces_per_token 5 \
    --model_dir ${base_dir}/ckpts \
    --ckpt_id "model" \
    --detect_vocab_path "./data/vocabulary/d_tags.txt" \
    --correct_vocab_path "./data/vocabulary/labels_vi.txt" \
    --pretrained_transformer_path "vinai/phobert-base" \
    --input_path "./datatest/source_test.txt" \
    --out_path "result/yaclc-minimal_testA.preds" \
    --special_tokens_fix 1 \
    --detokenize 1 \
    --amp

python test.py
