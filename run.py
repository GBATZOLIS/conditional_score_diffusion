from job_master.run_configs import run_configs

# configs = [
#     'configs/ksphere/robustness/0_005.py',
#     'configs/ksphere/robustness/0_01.py',
#     'configs/ksphere/robustness/0_015.py',
#     'configs/ksphere/robustness/0_1.py',
#     'configs/ksphere/robustness/1.py',
# ]

# configs = [
#     'configs/dimension_estimation/ksphere/N_1/non_uniform_1.py',
#     'configs/dimension_estimation/ksphere/N_1/non_uniform_075.py',
#     'configs/dimension_estimation/ksphere/N_1/non_uniform_05.py',
#     'configs/dimension_estimation/ksphere/N_1/non_uniform_025.py',
#     'configs/dimension_estimation/ksphere/2d_toy/vesde.py'
# ]


# configs = ['configs/daniel/daniel.py']
configs=['configs/dimension_estimation/mammoth/vesde.py']

run_configs(configs, partition='ampere', time='36:00:00')
