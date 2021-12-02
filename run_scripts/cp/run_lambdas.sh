#!/bin/bash
python main.py --config configs/jan/holiday/circles/cp/LAMBDA_0_1.py \
               --log_path ./potential
python main.py --config configs/jan/holiday/circles/cp/LAMBDA_0_01.py \
               --log_path ./potential
python main.py --config configs/jan/holiday/circles/cp/LAMBDA_inf.py \
               --log_path ./potential