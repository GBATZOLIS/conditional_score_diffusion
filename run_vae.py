import os
import subprocess
import re
from job_master.utils import create_mainsh
from configs.utils import read_config
from pathlib import Path

configs = [
            'configs/VAE/celebA.py',
            'configs/VAE/cifar.py',
        ]
           

repo_path=Path('.').absolute()

for config_path in configs:

    config = read_config(config_path)
    output_path = os.path.join(repo_path, 'slurm_files')
    main_sh_path = os.path.join(output_path ,f'main.sh')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    create_mainsh(main_sh_path = main_sh_path,
                  file_path='vae_main .py',
                  args_dict={'config': config_path},
                  job_name=f'VAE_{config.data.dataset}_kl_{config.model.kl_weight}',
                  partition='ampere',
                  account='CIA-DAMTP-SL2-GPU',
                  repo_path=repo_path,
                  mail='js2164@cam.ac.uk',
                  time='36:00:00'
                  )

    cmds = [f'cd {output_path}; sbatch main.sh']
    result = subprocess.run(cmds, capture_output=True, text=True, shell=True, executable='/bin/bash')
    out = result.stdout
    id = re.findall("\d+", out)[0]
    print(f'Job submitted as {id}')
    os.remove(main_sh_path)