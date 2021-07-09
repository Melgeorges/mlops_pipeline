from sklearn import svm
from sklearn.model_selection import cross_val_score
import pickle
from loguru import logger
import pandas as pd

SCORE_MIN = 90


def get_data(file):
    df = pd.read_csv(file, header=0)
    X = df.loc[:, df.columns != 'output']
    Y = df["output"]

    return X, Y


def get_params(file="input/params.json"):

    param_gamma = 0.001
    param_C = 100.
    return param_gamma, param_C


def make_model(X, Y, param_gamma, param_C):
    model = svm.SVC(gamma=param_gamma, C=param_C)
    scores = cross_val_score(model, X, Y, cv=5)  # cv est le nombre de découpages à réaliser
    score = scores.mean()

    return model, score


def set_to_prod(model, score, SCORE_MIN):
    if score > SCORE_MIN:
        pickle.dump(model, open("production_model", 'wb'))
    else:
        logger.warning(f"The model score is {score}, not enough to be sent into production")
