git add .
git commit -m 'update'
git push
ssh hpc 'cd /home/js2164/rds/hpc-work/repos/score_sde_pytorch; git pull'