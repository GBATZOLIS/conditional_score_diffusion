import os 
import sys
from pathlib import Path
repo_path=Path('.').absolute().parent.parent.absolute()
print(f'Moving cwd to {repo_path}')
os.chdir(repo_path)
sys.path.append(repo_path)