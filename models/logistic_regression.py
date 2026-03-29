import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from shared.preprocess import build_preprocessor

# The model name
MODEL_NAME = "logistic_regression"

def load_split_data():
    X_train = pd.read_csv("../data/split/X_train.csv")
    X_test = pd.read_csv("../data/split/X_test.csv")
    y_train = pd.read_csv("../data/split/y_train.csv").squeeze("columns")
    y_test = pd.read_csv("../data/split/y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test

def build_model(processor):
    # Build a logistic regression model with a consistent pipeline

    logistic_regression_pipeline_model = Pipeline([
        ("preprocessor", build_preprocessor(processor)),
        ("classifier", LogisticRegression())
    ])

    return logistic_regression_pipeline_model

