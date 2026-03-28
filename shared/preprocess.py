import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Sleep Disorder"

def clean_raw_data(df):
    # Make a copy for comparison
    cleaned_data = df.copy()

    cleaned_data[TARGET_COLUMN] = cleaned_data[TARGET_COLUMN].fillna("None").astype(str).str.strip()

    # Standardize
    cleaned_data["BMI Category"] = cleaned_data["BMI Category"].replace({"Normal Weight": "Normal"})

    # Split blood pressure into 2 columns
    bp_split = cleaned_data["Blood Pressure"].astype(str).str.split("/", n=1, expand=True)

    cleaned_data["Systolic BP"] = pd.to_numeric(bp_split[0], errors="coerce")
    cleaned_data["Diastolic BP"] = pd.to_numeric(bp_split[1], errors="coerce")

    # Drop used columns
    cleaned_health_data = cleaned_data.drop(columns=["Blood Pressure", "Person ID"])

    return cleaned_health_data

def get_feature_columns(cleaned_health_data):
    numeric_cols = cleaned_health_data.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = cleaned_health_data.select_dtypes(exclude=["number"]).columns.tolist()

    return numeric_cols, categorical_cols

def build_processor(X_train):
    numeric_cols, categorical_cols = get_feature_columns(X_train)

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer([("numeric", numeric_transformer, numeric_cols),
                                     ("categorical", categorical_transformer, categorical_cols)])

    return preprocessor




    
