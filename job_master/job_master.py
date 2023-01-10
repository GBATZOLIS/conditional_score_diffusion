import os
import subprocess
import re
import glob
import sys
sys.path.append('/home/js2164/rds/hpc-work/repos/score_sde_pytorch/')
from configs.utils import read_config



class Job():

    def __init__(self):
        self.repo_path = '/home/js2164/rds/hpc-work/repos/score_sde_pytorch/'
        self.job_dir = None
        self.id = None
        self.config = None
        self.mode = 'train'
        self.log = None
        self.name = None
        self.checkpoint_path = None
        self.count = 0

    def read_from_file(self, path):
        self.job_dir = path
        with open(path + '/job', 'r') as job_file:
            lines=job_file.readlines()
            # add trailing endline
            if lines[-1][-2:-1] != '\n':
                lines[-1] = lines[-1] + '\n'
            lines = ''.join(lines)
            self.config = re.findall("config=([\w|/ | \.]+)", lines)[0]
            if len(re.findall("mode=([\w|/ | \.]+)", lines)) > 0:
                self.mode = re.findall("mode=([\w|/ | \.]+)", lines)[0]
            config = read_config(self.config)
            self.log = config.logging.log_path #re.findall("log=([\w|/ | \.]+)", lines)[0]
            self.name = config.logging.log_name #re.findall("name=([\w|/ | \.]+)", lines)[0]
            self.count = re.findall("count=([\w|/ | \.]+)", lines)[0]
            self.count = int(self.count)

            checkpoint_path = re.findall("checkpoint=([\w|/ | \.]+)", lines)
            self.checkpoint_path = None if checkpoint_path == [] else checkpoint_path[0]

            id = re.findall("checkpoint=([\w|/ | \.]+)", lines)
            self.id = None if id == [] else id[0]

    def update_job_file(self):
        lines = [
            'config='+self.config+'\n',
            #'log='+self.log+'\n',
            #'name='+self.name+'\n',
            'count='+str(self.count)+'\n'
        ]
        if self.checkpoint_path is not None:
            lines = lines + ['checkpoint='+self.checkpoint_path+'\n']
        if self.id is not None:
            lines = lines + ['id='+self.id+'\n']

        with open(self.job_dir+'/job','w') as job_file:
            job_file.writelines(lines)
            
class JobMaster():
    
    def __init__(self):
        self.repo_path = '/home/js2164/rds/hpc-work/repos/score_sde_pytorch/'

    def submit(self, job):   
        self.create_wilkes(job)        
        cmds = [f'cd {job.job_dir}; sbatch main.sh']
        result = subprocess.run(cmds, capture_output=True, text=True, shell=True, executable='/bin/bash')
        out = result.stdout
        id = re.findall("\d+", out)[0]
        return id

    def run_job(self, job):
        if job.count <= 0:
            return None
        if job.id is None:
            # Starting fresh job
            id = self.submit(job)
        else:
            # Resuming job
            if self.check_if_running(job): 
                return None
            else:                 # if not running 
                job.checkpoint_path = self.get_checkpoint_path(job)
                id = self.submit(job)

        job.id = id
        job.count -= 1
        job.update_job_file()

    def get_checkpoint_path(self, job):
        dir_path = os.path.join(self.repo_path, job.log, job.name,'checkpoints') 
        chpt_path = glob.glob(dir_path)[-1]
        return chpt_path


    def check_if_running(self, job):
        cmd = 'squeue -u js2164 | grep ' + str(job.id)
        result = subprocess.run([cmd], capture_output=True, text=True, shell=True)
        return False if result.stdout == '' else True


    def create_wilkes(self, job):
        main_sh_path = job.job_dir + '/main.sh'
        
        with open(main_sh_path,'w') as main_sh:
            L = [
                '#!/bin/bash \n',

                '#! Name of the job: \n',
                f'#SBATCH -J {job.name} \n',
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
                '#SBATCH -p ampere \n',
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

                f'cd {job.repo_path} \n',

                f'python main.py --config {job.config} \\ \n',
                            f' --mode {job.mode} \\ \n',
                            #f' --checkpoint_path {job.checkpoint_path} \n' if job.checkpoint_path is not None else '',
                            #f'--log_path {job.log} \\ \n',
                            #f'--log_name {job.name} \\ \n',
            ]

            main_sh.writelines(L)