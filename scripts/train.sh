#!/usr/bin/env bash

cd ..
python main.py  --config configs/jan/holiday/unconditional_generation.py \
                --mode train \
                --log_path ./ \
