import json
import joblib
from datetime import datetime
from pathlib import Path


def save_model(model, path="models/churn_model.joblib"):
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def save_metadata(metrics: dict, path="models/model_metadata.json"):
    Path("models").mkdir(exist_ok=True)
    metrics["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metadata saved → {path}")
