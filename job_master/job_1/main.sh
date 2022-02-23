#!/bin/bash 
module unload miniconda/3 
module load cuda/11.4 
source /home/js2164/.bashrc 
conda activate score_sde 
cd /home/js2164/rds/hpc-work/repos/score_sde_pytorch/
python main.py --config configs/jan/celebA/hpc/run.py --log_path logs/celebA/potential --log_name snr