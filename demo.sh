#!/bin/bash

set -e

UGC_FILE="data/draft/ugc.txt"
STD_FILE="data/draft/std.txt"
OUTPUT_DIR="outputs/draft"

mkdir -p $OUTPUT_DIR

MODEL[0]=$LASER/models/laser2.pt

MODEL[1]=models/RoLASER/rolaser.pt

MODEL[2]=models/c-RoLASER/c-rolaser.pt

for i in {0..2}
do
    echo "Evaluating ${MODEL[$i]}"

    python evaluation/cos_dist.py \
        -m ${MODEL[$i]} \
        --ugc-file $UGC_FILE \
        --std-file $STD_FILE \
        -o $OUTPUT_DIR \
        --verbose | tee -a $OUTPUT_DIR/outputs.log

done

echo Averaging and plotting all scores...

python evaluation/avg_cos_dist.py -o $OUTPUT_DIR

echo Done...