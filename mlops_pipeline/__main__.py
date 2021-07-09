import os
from datetime import datetime, timedelta
from time import sleep
from mlops_pipeline import move_data
from mlops_pipeline.mlflow_follow_up import run_model
from mlops_pipeline.modelisation import set_to_prod
import mlflow
from loguru import logger

RUN_TIME = 5
SCORE_MIN = 0.5


run_time = timedelta(minutes=RUN_TIME)
start_time = datetime.now()
current_time = start_time
mlflow.set_experiment("/heart")
print("a")

while run_time > current_time-start_time:

    directory = "input/data/"
    files = os.listdir(directory)
    logger.info(f"files to process : {files}")

    if files:
        for file in files:
            data_path = os.path.join(directory, file)
            model, score = run_model(data_path)
            set_to_prod(model, score, SCORE_MIN)
            move_data(data_path)

    else:
        sleep(10)

    files = []
    current_time = datetime.now()
    logger.info(f"{current_time-start_time}")
