from mlflow import log_metric, log_param
import mlflow.sklearn
from mlops_pipeline.modelisation import get_data, get_params, make_model, preprocess_data


def run_model(file):

    mlflow.start_run(run_name=file)

    # Get data - log file name
    X, Y = get_data(file)
    X = preprocess_data(X)

    # Get params - log them
    param_gamma, param_C = get_params()
    log_param("gamma", param_gamma)
    log_param("C", param_C)

    # Run model
    model, score = make_model(X, Y, param_gamma, param_C)
    log_metric("score", score)

    # Save model into mlflow / make it accesible
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

    return model, score




