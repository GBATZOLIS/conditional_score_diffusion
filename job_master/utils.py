import importlib
import re
from pathlib import Path
REPO_PATH=Path('.').absolute().parent.absolute()

def get_account_for_partition(partition):
    if partition == 'pascal':
        account = 'CIA-DAMTP-SL2-GPU'
    elif partition == 'ampere':
        account = 'CIA-DAMTP-SL2-GPU'
    return account

def read_config(config_path):
    module_name = re.findall('configs/[\w|/ | \.]+.py', config_path)[0][:-3].replace('/','.')
    module = importlib.import_module(module_name)
    config = module.get_config()
    return config

def create_mainsh(main_sh_path,
                  file_path,
                  args_dict,
                  job_name, 
                  partition,
                  mail,
                  repo_path=REPO_PATH,
                  account=None
                  ):
    if account is None:
        account = get_account_for_partition(partition)

    args_string = ' '
    for key, value in args_dict.items():
        args_string += f'--{key}'
        args_string += f' {value} '


    with open(main_sh_path,'w') as main_sh:
        L = [
            '#!/bin/bash \n',

            '#! Name of the job: \n',
            f'#SBATCH -J {job_name} \n',
            '#SBATCH -o JOB%j.out # File to which STDOUT will be written \n',
            '#SBATCH -e JOB%j.out # File to which STDERR will be written \n',

            '#! Which project should be charged (NB Wilkes2 projects end in \'-GPU\'): \n',
            f'#SBATCH --account {account} \n',

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
            f'#SBATCH --mail-user={mail} \n',

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
            ' \n',

            f'cd {repo_path} \n',

            f'python {file_path} {args_string} \\ \n',
        ]

        main_sh.writelines(L)
