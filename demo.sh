#!/bin/bash

set -e

UGC_FILE="data/demo_ugc.txt"
STD_FILE="data/demo_std.txt"
OUTPUT_DIR="outputs/"

MODELS="LASER RoLASER c-RoLASER"

for MODEL in $MODELS
do
    MODEL_DIR=models/${MODEL}

    python evaluation/rolaser.py -m $MODEL_DIR --ugc-file $UGC_FILE --std-file $STD_FILE -o $OUTPUT_DIR

    cat outputs/outputs_${MODEL}.txt | tee -a outputs/outputs.log
done