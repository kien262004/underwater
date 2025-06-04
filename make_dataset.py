import os
import shutil


root = './Under_Water/Datasets/EUVP/Paired/'
tasks = os.listdir(root)
input_dir = 'trainA'
target_dir = 'trainB'
target_task = 'total'

target_task_path = os.path.join(root, target_task)
os.makedirs(target_task_path, exist_ok=True)
os.makedirs(os.path.join(target_task_path, input_dir), exist_ok=True)
os.makedirs(os.path.join(target_task_path, target_dir), exist_ok=True)


for task in tasks:
    inp_path = os.path.join(root, task, input_dir)
    tar_path = os.path.join(root, task, target_dir)
    filenames = os.listdir(inp_path)
    for filename in filenames:
        shutil.copy(os.path.join(inp_path, filename), os.path.join(target_task_path, input_dir))
        shutil.copy(os.path.join(tar_path, filename), os.path.join(target_task_path, target_dir))

    
    