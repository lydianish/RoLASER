#!/bin/bash

set -e

if [ -z ${LASER} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit 1
fi

UGC_FILE=/home/lnishimw/scratch/datasets/rocsmt/test/norm.en.test
#UGC_FILE="data/demo_ugc.txt"
STD_FILE=/home/lnishimw/scratch/datasets/rocsmt/test/raw.en.test 
#STD_FILE="data/demo_std.txt"
OUTPUT_DIR="outputs"

mkdir -p $OUTPUT_DIR

MODEL_NAME[0]="LASER"
MODEL_DIR[0]=$LASER/models
TOKENIZER[0]="spm"

MODEL_NAME[1]="RoLASER"
MODEL_DIR[1]=models/RoLASER
TOKENIZER[1]="roberta"

MODEL_NAME[2]="c-RoLASER"
MODEL_DIR[2]=models/c-RoLASER
TOKENIZER[2]="char"

for i in {0..0}
do
    echo "Evaluating ${MODEL_NAME[$i]}..."
    python evaluation/eval_files.py -m ${MODEL_NAME[$i]} \
        -d ${MODEL_DIR[$i]} \
        --ugc-file $UGC_FILE \
        --std-file $STD_FILE \
        -o $OUTPUT_DIR | tee -a $OUTPUT_DIR/outputs.log

    # python evaluation/cos_dist.py -m ${MODEL_NAME[$i]} \
    #     -d ${MODEL_DIR[$i]} \
    #     -t ${TOKENIZER[$i]} \
    #     --ugc-file $UGC_FILE \
    #     --std-file $STD_FILE \
    #     -o $OUTPUT_DIR \
    #     --verbose | tee -a $OUTPUT_DIR/outputs.log

done

echo Done...