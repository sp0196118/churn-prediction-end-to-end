import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

from data_preprocessing import load_data, build_preprocessing_pipeline, split
from utils import save_model, save_metadata


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():

    print("📥 Loading data...")
    df = load_data(DATA_PATH)

    print("🛠 Building preprocessing pipeline...")
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(df)

    X_train, X_test, y_train, y_test = split(df)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor),
                         ("model", model)])

    print("🚀 Training model...")
    clf.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc:.3f}")

    save_model(clf)

    save_metadata(
        {
            "algorithm": "RandomForestClassifier",
            "roc_auc": roc,
            "features_numeric": num_cols,
            "features_categorical": cat_cols,
        }
    )


if __name__ == "__main__":
    main()

