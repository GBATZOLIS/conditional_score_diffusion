import os
import subprocess
import re
import sys
repo_path = '/home/js2164/rds/hpc-work/repos/score_sde_pytorch/'
sys.path.append(repo_path)
from configs.utils import read_config


tmp_path = os.path.join(repo_path, 'job_master', 'tmp')
main_sh_path = os.path.join(tmp_path, 'main.sh')

def create_mainsh(config_path, mode = 'train', partition='ampere'):

        config = read_config(config_path)
        name = config.logging.log_name
        log_path = config.logging.log_path

        
        with open(main_sh_path,'w') as main_sh:
            L = [
                '#!/bin/bash \n',

                '#! Name of the job: \n',
                f'#SBATCH -J {name} \n',
                '#SBATCH -o JOB%j.out # File to which STDOUT will be written \n',
                '#SBATCH -e JOB%j.out # File to which STDERR will be written \n',

                '#! Which project should be charged (NB Wilkes2 projects end in \'-GPU\'): \n',
                '#SBATCH --account SCHOENLIEB-SL3-GPU \n',

                '#! How many whole nodes should be allocated? \n',
                '#SBATCH --nodes=1 \n',

                '#! How many (MPI) tasks will there be in total? \n',
                '#! Note probably this should not exceed the total number of GPUs in use. \n',
                '#SBATCH --ntasks=1 \n',

                '#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1). \n',
                '#! Note that the job submission script will enforce no more than 32 cpus per GPU. \n',
                '#SBATCH --gres=gpu:1 \n',

                '#! How much wallclock time will be required? \n',
                '#SBATCH --time=12:00:00 \n',

                '#! What types of email messages do you wish to receive? \n',
                '#SBATCH --mail-type=begin        # send email when job begins \n',
                '#SBATCH --mail-type=end \n',
                '#SBATCH --mail-user=js2164@cam.ac.uk \n',

                '#! Do not change: \n',
                f'#SBATCH -p {partition} \n',
                ' \n',

                '. /etc/profile.d/modules.sh                # Leave this line (enables the module command) \n',
                'module purge                               # Removes all modules still loaded \n',
                'module load rhel8/default-amp              # REQUIRED - loads the basic environment \n',

                'module unload miniconda/3 \n',
                'module load cuda/11.4 \n',

                'module list \n',

                'nvidia-smi \n',

                'source /home/js2164/.bashrc \n',
                'conda activate score_sde \n',

                'REPO=/rds/user/js2164/hpc-work/repos/score_sde_pytorch/ \n',
                ' \n',

                f'cd {repo_path} \n',

                f'python main.py --config {config_path} \\ \n',
                            f' --mode {mode} \\ \n',
            ]

            main_sh.writelines(L)


def run_config(config):
    create_mainsh(config)
    cmds = [f'cd {tmp_path}; sbatch main.sh']
    result = subprocess.run(cmds, capture_output=True, text=True, shell=True, executable='/bin/bash')
    out = result.stdout
    id = re.findall("\d+", out)[0]
    print(f'Job submitted as {id}')
    os.remove(main_sh_path)

def run_configs(configs):
    for config in configs:
        run_config(config)