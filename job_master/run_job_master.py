#%%
import job_master
master = job_master.JobMaster()
# %%
job = job_master.Job()
job.read_from_file('/home/js2164/rds/hpc-work/repos/score_sde_pytorch/job_master/job_1/')
#%%
master.run_job(job)

# %%
