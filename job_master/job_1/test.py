import os

path = "/home/js2164/rds/hpc-work/repos/score_sde_pytorch/job_master/job_1/main.sh"
r = os.system(f"sbatch {path}")
print('result', r)
