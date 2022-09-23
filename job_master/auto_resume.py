import subprocess
import os

path = '/rds/user/js2164/hpc-work/repos/score_sde_pytorch/'
job_dir = path + 'wilkes_scripts/job7/'


os.chdir(job_dir)
cmds = ['sbatch ' + job_dir + 'wilkes3_script']
result = subprocess.run(cmds, capture_output=True, text=True, shell=True, executable='/bin/bash')
out = result.stdout
print(result)