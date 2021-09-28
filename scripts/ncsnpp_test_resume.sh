#!/usr/bin/env bash

cd ..
python main.py  --config configs/jan/holiday/ncsnpp_tets.py \
                --mode train \
                --log_path ./ \
                --checkpoint_path /store/CIA/js2164/repos/diffusion_score/lightning_logs/version_12/checkpoints/epoch=43-step=142647.ckpt
