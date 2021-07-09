from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import pickle
from loguru import logger
import pandas as pd
import json

SCORE_MIN = 90


def get_data(file):
    df = pd.read_csv(file, header=0)
    X = df.loc[:, df.columns != 'output']
    Y = df["output"]

    return X, Y


def get_params(file="input/params/params.json"):

    with open(file) as f:
        params = json.load(f)

    param_gamma = params['svm']['param_gamma']
    param_C = params['svm']['param_C']
    return param_gamma, param_C


def preprocess_data(X):
    return X


def split_data(X, Y):
    seed = 42
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
    return X_train, X_test, Y_train, Y_test, seed


def make_model(X, Y, param_gamma, param_C):
    clf = svm.SVC(gamma=param_gamma, C=param_C)
    scores = cross_val_score(clf, X, Y, cv=5)  # cv est le nombre de découpages à réaliser
    score_cv = scores.mean()

    model = clf.fit(X, Y)

    return model, score_cv


def get_scores(model, X_test, Y_test):

    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    return accuracy


def set_to_prod(model, score, SCORE_MIN):
    if score > SCORE_MIN:
        pickle.dump(model, open("production_model/production.p", 'wb'))
        return True
    else:
        logger.warning(f"The model score is {score}, not enough to be sent into production")
        return False
