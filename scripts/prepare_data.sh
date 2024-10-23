#!/bin/bash
SUBSET="train-text"
SOURCE="/content/drive/MyDrive/core/data_train/Text_Text/source_train.txt"
TARGET="/content/drive/MyDrive/core/data_train/Text_Text/target_train.txt"
OUTPUT="./gec_private_train_data/${SUBSET}.edits"
python utils/preprocess_data.py -s $SOURCE -t $TARGET -o $OUTPUT

SUBSET1="test-text"
SOURCE1="/content/drive/MyDrive/core/data_train/Text_Text/source_val.txt"
TARGET1="/content/drive/MyDrive/core/data_train/Text_Text/target_val.txt"
OUTPUT1="./gec_private_train_data/${SUBSET1}.edits"
python utils/preprocess_data.py -s $SOURCE1 -t $TARGET1 -o $OUTPUT1