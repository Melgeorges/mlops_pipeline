import os
from datetime import datetime, timedelta
from time import sleep
from mlops_pipeline import move_data
from mlops_pipeline.mlflow_follow_up import run_model_safe
from mlops_pipeline.modelisation import set_to_prod
from mlops_pipeline.monitor import launch_monitoring
import mlflow
from loguru import logger

RUN_TIME = 5
SCORE_MIN = 0.5
EXPERIMENT_NAME = "heart"
PROD = True


run_time = timedelta(minutes=RUN_TIME)
start_time = datetime.now()
current_time = start_time
mlflow.set_experiment(f"/{EXPERIMENT_NAME}")

while run_time > current_time-start_time:

    directory = "input/data/"
    files = os.listdir(directory)
    logger.info(f"files to process : {files}")

    if files:
        for file in files:
            data_path = os.path.join(directory, file)
            # It would actually be nice to test the previous model on new data, to see if on track or not through time
            model, score_cv, accuracy = run_model_safe(data_path)
            if model and PROD:
                is_prod = set_to_prod(model, accuracy, SCORE_MIN)
                launch_monitoring(EXPERIMENT_NAME, is_prod)
                move_data(data_path)
    else:
        sleep(10)

    files = []
    current_time = datetime.now()
    logger.info(f"{current_time-start_time}")
