#!/bin/bash
base_dir="/content/drive/MyDrive/Gector/fast-gector"
mkdir result
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
    --input_path "/content/drive/MyDrive/Gector/fast-gector/source_val.txt" \
    --out_path "result/yaclc-minimal_testA.preds" \
    --special_tokens_fix 1 \
    --detokenize 1 \
    --amp
