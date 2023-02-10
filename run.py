from job_master.run_configs import run_configs

configs = [
    'configs/ksphere/robustness/0.py',
    'configs/ksphere/robustness/0_25.py',
    'configs/ksphere/robustness/0_5.py',
    'configs/ksphere/robustness/0_75.py',
    'configs/ksphere/robustness/1.py',
]

run_configs(configs)
