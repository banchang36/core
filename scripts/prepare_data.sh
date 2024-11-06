#!/bin/bash
SUBSET="train-stage2"
SOURCE="source_train.txt"
TARGET="target_train.txt"
OUTPUT="./gec_private_train_data/${SUBSET}.edits"
python utils/preprocess_data.py -s $SOURCE -t $TARGET -o $OUTPUT

SUBSET1="test"
SOURCE1="source_val.txt"
TARGET1="target_val.txt"
OUTPUT1="./gec_private_train_data/${SUBSET1}.edits"
python utils/preprocess_data.py -s $SOURCE1 -t $TARGET1 -o $OUTPUT1
