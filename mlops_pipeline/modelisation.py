from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score


def get_data(file):
    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target

    return X, Y


def get_params(file="input_params"):

    param_gamma = 0.001
    param_C = 100.
    return param_gamma, param_C


def make_model(X, Y, param_gamma, param_C):
    model = svm.SVC(gamma=param_gamma, C=param_C)
    scores = cross_val_score(model, X, Y, cv=5)  # cv est le nombre de découpages à réaliser
    score = scores.mean()

    return model, score