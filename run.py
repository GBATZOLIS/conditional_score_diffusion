from job_master.run_configs import run_configs

# configs = [
#     'configs/ksphere/robustness/0_005.py',
#     'configs/ksphere/robustness/0_01.py',
#     'configs/ksphere/robustness/0_015.py',
#     'configs/ksphere/robustness/0_1.py',
#     'configs/ksphere/robustness/1.py',
# ]

configs = [
     'configs/ksphere/N_1/spectrum_in_training.py',
#     'configs/ksphere/sample_complexity/100.py',
#     'configs/ksphere/sample_complexity/10000.py',
#     'configs/ksphere/sample_complexity/100000.py',
#     'configs/ksphere/sample_complexity/1000000.py',
 ]


# configs = ['configs/daniel/daniel.py']

run_configs(configs, partition='ampere')
