from mlflow import log_metric, log_param, log_artifacts
import mlflow.sklearn
from mlops_pipeline.modelisation import get_data, get_params, make_model, preprocess_data, split_data, get_scores
from mlops_pipeline import get_commit


def run_model_safe(file):
    try:
        model, score_cv, accuracy = run_model(file)

    except Exception as e:
        with open("outputs/exception.txt", "w") as f:
            f.write(f"Got an error message on the run : {e}")
        log_artifacts("outputs")
        model = None
        score_cv = None
        accuracy = None
        mlflow.end_run()

    return model, score_cv, accuracy


def run_model(file):

    mlflow.start_run(run_name=file)

    # Get data - log file name
    X, Y = get_data(file)
    X = preprocess_data(X)
    X_train, X_test, Y_train, Y_test, seed = split_data(X, Y)
    log_param("seed", seed)

    # Get params - log them
    param_gamma, param_C = get_params()
    log_param("gamma", param_gamma)
    log_param("C", param_C)

    # Run model
    model, score_cv = make_model(X_train, Y_train, param_gamma, param_C)
    log_metric("score_cv", score_cv)

    accuracy = get_scores(model, X_test, Y_test)
    log_metric("accuracy", accuracy)

    # Save model into mlflow / make it accesible
    mlflow.sklearn.log_model(model, "model")

    r = get_commit('/home/melanie/Documents/code/mlops_pipeline')
    with open("outputs/git_info.txt", "w") as f:
        f.write(f"code used is project MLOPS_PIPELINE, commit {r}")

    with open("outputs/test.txt", "w") as f:
        f.write("Any kind of log on the the run should end up here to be easily accessible to any one, images, ...")
    log_artifacts("outputs")

    mlflow.end_run()

    return model, score_cv, accuracy




