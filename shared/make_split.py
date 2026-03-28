import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    health_data = pd.read_csv('../data/cleaned/cleaned.csv')
    return health_data

def split_data(health_data):
    X = health_data.drop(columns = "Sleep Disorder")
    y = health_data['Sleep Disorder']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    return X_train, X_test, y_train, y_test

def save_to_csv(X_train, X_test, y_train, y_test):
    X_train.to_csv('../data/split/X_train.csv', index = False)
    X_test.to_csv('../data/split/X_test.csv', index = False)
    y_train.to_csv('../data/split/y_train.csv', index = False)
    y_test.to_csv('../data/split/y_test.csv', index = False)



