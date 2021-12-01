#!/bin/bash
python main.py --config configs/jan/holiday/circles/cp/LAMBDA_0.py \
               --log_path ./potential
python main.py --config configs/jan/holiday/circles/cp/LAMBDA_1.py \
               --log_path ./potential
python main.py --config configs/jan/holiday/circles/cp/LAMBDA_10.py \
               --log_path ./potential