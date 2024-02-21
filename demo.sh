#!/bin/bash

set -e

UGC_FILE="data/demo_ugc.txt"
STD_FILE="data/demo_std.txt"
OUTPUT_DIR="outputs"

mkdir -p $OUTPUT_DIR

MODEL_DIR[0]="models/LASER"
TOKENIZER[0]="spm"

MODEL_DIR[1]="models/RoLASER"
TOKENIZER[1]="roberta"

MODEL_DIR[2]="models/c-RoLASER"
TOKENIZER[2]="char"

for i in {0..2}
do

    python evaluation/cos_dist.py -m ${MODEL_DIR[$i]} \
        -t ${TOKENIZER[$i]} \
        --ugc-file $UGC_FILE \
        --std-file $STD_FILE \
        -o $OUTPUT_DIR \
        --verbose | tee -a $OUTPUT_DIR/outputs.log

done