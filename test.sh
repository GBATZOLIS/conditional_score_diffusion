conda activate score_sde

REPO=/home/js2164/jan/repos/diffusion_score/


CONFIG=configs/jan/circles/experiments/potential.py


LOG=logs/circles/fokker_planck


NAME=potential_ve

cd $REPO

python main.py --config $CONFIG \
               --log_path $LOG \
               --log_name $NAME \
       
               