#!/bin/bash
#! Name of the job:
#SBATCH -J MY_JOB
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j.out # File to which STDERR will be written
#! Which project should be charged (NB Wilkes2 projects end in \-GPU\):
#SBATCH --account SCHONLIEB-SL3-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=12:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end
#SBATCH --mail-user=td314@bath.ac.uk
#! Do not change:
#SBATCH -p pascal
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module unload miniconda/3
module load cuda/11.4
module list
nvidia-smi
source /home/td491/.bashrc
conda activate scoresde_env
cd /home/td491/rds/hpc-work/conditional_score_diffusion
python main.py --config configs/jan/circles/fokker_planck/0_small.py --mode train