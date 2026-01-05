import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


TARGET = "Churn"


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from csv."""
    return pd.read_csv(path)


def build_preprocessing_pipeline(df: pd.DataFrame):
    """Create preprocessing pipeline for numeric + categorical features."""

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if TARGET in categorical_cols:
        categorical_cols.remove(TARGET)

    numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def split(df: pd.DataFrame):
    """Split into train and test."""

    X = df.drop(columns=[TARGET])
    y = df[TARGET].apply(lambda x: 1 if x == "Yes" else 0)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
