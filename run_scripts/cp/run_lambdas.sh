#!/bin/bash

COMMAND1="conda activate score_sde; sleep 10; python main.py --config /home/js2164/jan/repos/diffusion_score/configs/jan/holiday/circles/cp/LAMBDA_0.py \
               --log_path ./test_logs; sleep 10"
COMMAND2="conda activate score_sde; sleep 10; python main.py --config /home/js2164/jan/repos/diffusion_score/configs/jan/holiday/circles/cp/LAMBDA_1.py \
               --log_path ./test_logs; sleep 10"
COMMAND3="conda activate score_sde; sleep 10; python main.py --config /home/js2164/jan/repos/diffusion_score/configs/jan/holiday/circles/cp/LAMBDA_10.py \
               --log_path ./test_logs; sleep 10"
COMmAND4="conda activate score_sde; sleep 10; python main.py --config /home/js2164/jan/repos/diffusion_score/configs/jan/holiday/circles/cp/LAMBDA_100.py \
               --log_path ./test_logs; sleep 10"

tmux attach -t test ';' new-window "zsh" ';'  send-keys "${COMMAND1}" ENTER ';' \ 
                        new-window "zsh" ';'  send-keys "${COMMAND2}" ENTER ';' \
                        new-window "zsh" ';'  send-keys "${COMMAND3}" ENTER ';' \
                        new-window "zsh" ';'  send-keys "${COMMAND4}" ENTER 