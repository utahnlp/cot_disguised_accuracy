#!/bin/bash

NUM_EXAMPLES=1000
OUT_DIR=./data

for digits in 2 3; do
    for operands in 2 4 8 16; do
        python src/addition.py \
            --out_dir $OUT_DIR \
            --num_examples $NUM_EXAMPLES \
            --digits $digits \
            --operands $operands
    done
done
