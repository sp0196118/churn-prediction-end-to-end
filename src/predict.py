import pandas as pd
import joblib


MODEL_PATH = "models/churn_model.joblib"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_from_csv(csv_path):
    model = load_model()
    df = pd.read_csv(csv_path)
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    results = df.copy()
    results["churn_prediction"] = preds
    results["probability"] = probs

    return results


if __name__ == "__main__":
    sample_file = "data/sample.csv"   # replace with your test file
    results = predict_from_csv(sample_file)
    print(results.head())
