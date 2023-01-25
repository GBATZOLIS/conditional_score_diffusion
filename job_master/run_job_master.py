import job_master
master = job_master.JobMaster()

job1 = job_master.Job()
job1.read_from_file('/home/js2164/rds/hpc-work/repos/score_sde_pytorch/job_master/job_1/')
master.run_job(job1)

job2 = job_master.Job()
job2.read_from_file('/home/js2164/rds/hpc-work/repos/score_sde_pytorch/job_master/job_2/')
master.run_job(job2)

job3 = job_master.Job()
job3.read_from_file('/home/js2164/rds/hpc-work/repos/score_sde_pytorch/job_master/job_3/')
master.run_job(job3)

job4 = job_master.Job()
job4.read_from_file('/home/js2164/rds/hpc-work/repos/score_sde_pytorch/job_master/job_4/')
master.run_job(job4)

job5 = job_master.Job()
job5.read_from_file('/home/js2164/rds/hpc-work/repos/score_sde_pytorch/job_master/job_5/')
master.run_job(job5)