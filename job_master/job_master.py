import os
import subprocess
import re
import glob



class Job():

    def __init__(self):
        self.job_dir = None
        self.id = None
        self.config = None
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
            self.log = re.findall("log=([\w|/ | \.]+)", lines)[0]
            self.name = re.findall("name=([\w|/ | \.]+)", lines)[0]
            self.count = re.findall("count=([\w|/ | \.]+)", lines)[0]
            self.count = int(self.count)

            checkpoint_path = re.findall("checkpoint=([\w|/ | \.]+)", lines)
            self.checkpoint_path = None if checkpoint_path == [] else checkpoint_path[0]

            id = re.findall("checkpoint=([\w|/ | \.]+)", lines)
            self.id = None if id == [] else id[0]

    def update_job_file(self):
        lines = [
            'config='+self.config+'\n',
            'log='+self.log+'\n',
            'name='+self.name+'\n',
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
        self.wilkes_script_path = '/home/js2164/rds/hpc-work/repos/score_sde_pytorch/wilkes_scripts/job3/wilkes3_script'

    def submit(self, job):   
        self.create_main_and_wilkes(job)        
        cmds = ['cd ' + job.job_dir + '; sbatch wilkes_script']
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
            else: 
                # if not running 
                job.checkpoint_path = self.get_checkpoint_path(job)
                id = self.submit(job)

        job.id = id
        job.count -= 1
        job.update_job_file()

    def get_checkpoint_path(self, job):
        dir_path = self.repo_path + '/' + job.log + '/' + job.name + '/checkpoints'
        chpt_path = glob.glob(dir_path)[-1]
        return chpt_path


    def check_if_running(self, job):
        cmd = 'squeue -u js2164 | grep ' + str(job.id)
        result = subprocess.run([cmd], capture_output=True, text=True, shell=True)
        return False if result.stdout == '' else True


    def create_main_and_wilkes(self, job):
        main_sh_path = job.job_dir + '/main.sh'
        with open(main_sh_path,'w') as main_sh:
            L = ['#!/bin/bash \n',
            'module unload miniconda/3 \n',
            'module load cuda/11.4 \n',
            'source /home/js2164/.bashrc \n',
            'conda activate score_sde \n',
            'cd ' + self.repo_path + '\n'
            ]

            if job.checkpoint_path is not None:
                L = L + ['python main.py --config ' + job.config + ' --checkpoint_path' + job.checkpoint_path + ' --log_path ' + job.log + ' --log_name ' + job.name]
            else:
                 L = L + ['python main.py --config ' + job.config + ' --log_path ' + job.log + ' --log_name ' + job.name]

            main_sh.writelines(L)

        with open(self.wilkes_script_path, 'r') as wilkes_script:
            lines = wilkes_script.readlines()

        for i, line in enumerate(lines):
            if line[:11] == 'application':
                lines[i] = 'application=' + '\'' + main_sh_path + '\''

        with open(job.job_dir + '/wilkes_script', 'w') as wilkes_script:
            wilkes_script.writelines(lines)