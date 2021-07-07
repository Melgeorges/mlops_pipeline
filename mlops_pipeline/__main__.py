import os
from datetime import datetime, timedelta
from time import sleep
from mlops_pipeline.mlflow_follow_up import run_model

RUN_TIME = 30


run_time = timedelta(minutes=RUN_TIME)
start_time = datetime.now()
current_time = start_time


while run_time < current_time-start_time:

    directory = "input_data"
    files = os.listdir(directory)

    if files:
        for file in files:
            run_model(file)
            move_data(file)
    else:
        sleep(60)

    current_time = datetime.now()
