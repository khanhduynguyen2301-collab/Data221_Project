import pandas as pd

def load_split_data():
    X_train = pd.read_csv("../data/split/X_train.csv")
    X_test = pd.read_csv("../data/split/X_test.csv")
    y_train = pd.read_csv("../data/split/y_train.csv").squeeze("columns")
    y_test = pd.read_csv("../data/split/y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test