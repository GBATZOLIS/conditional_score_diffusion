#!/bin/bash

module unload miniconda/3
module load cuda/11.4

module list

nvidia-smi

source /home/js2164/.bashrc
conda activate score_sde

REPO=/home/js2164/jan/repos/diffusion_score/
#REPO=/home/js2164/jan/repos/diffusion_score/

CONFIG=configs/jan/circles/experiments/potential.py
#configs/jan/circles/fokker_planck/fokker_planck_circles.py
#configs/jan/circles/vanilla_ve.py
#configs/jan/gaussian/vesde.py
#configs/jan/gaussian/fokker_planck_gauss.py
#configs/jan/circles/vanilla_ve.py
#configs/jan/gaussian/fokker_planck_gauss.py
#configs/jan/circles/vanilla_ve.py
#configs/jan/circles/fokker_planck/fokker_planck_circles.py
#configs/jan/gaussian/fokker_planck_gauss.py
#configs/jan/circles/fokker_planck/fokker_planck_circles.py
#configs/jan/crypto/default.py
#configs/jan/sine/default.py
#configs/jan/circles/fokker_planck/fokker_planck.py
#configs/jan/circles/vanilla_vp.py
#configs/jan/celebA/hpc.py
#configs/jan/circles/potential/circles_potential.py
#configs/jan/circles/curl_penalty/LAMBDA.py

LOG=logs/circles/fokker_planck
#logs/crypto
#logs/crypto
#logs/fokker_planck
#logs/circles_new
#logs/celebA

NAME=potential_ve

#CHECKPOINT=logs/curl_penalty_new/lightning_logs/LAMBDA_0/checkpoints/epoch=24999-step=1999999.ckpt

cd $REPO

python main.py --config $CONFIG \
               --log_path $LOG \
               --log_name $NAME
               #--checkpoint_path $CHECKPOINT