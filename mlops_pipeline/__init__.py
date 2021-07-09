import shutil


def move_data(file):
    archive_name = file.replace("data", "archive")
    shutil.move(file, archive_name)

