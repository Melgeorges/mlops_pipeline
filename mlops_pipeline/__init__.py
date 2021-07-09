import shutil
from pathlib import Path


def get_commit(repo_path):
    git_folder = Path(repo_path,'.git')
    head_name = Path(git_folder, 'HEAD').read_text().split('\n')[0].split(' ')[-1]
    head_ref = Path(git_folder,head_name)
    commit = head_ref.read_text().replace('\n','')
    return commit


def move_data(file):
    archive_name = file.replace("data", "archive")
    shutil.move(file, archive_name)

