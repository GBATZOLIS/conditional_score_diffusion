#!/bin/bash

cd ..

python main.py  --config /home/js2164/jan/repos/diffusion_score/configs/ve/celebahq_256_ncsnpp_continuous_jan_2.py \
                --eval_folder ./run_logs/ \
                --mode train \
                --workdir ./run_logs/

