#!/bin/bash

set -e

INPUT_FILE="data/demo/std.txt"

echo "Augmenting $INPUT_FILE"

python scripts/nlaugment.py \
    -i $INPUT_FILE \
    --seed 0 \
    --prob 0.1

echo Done...