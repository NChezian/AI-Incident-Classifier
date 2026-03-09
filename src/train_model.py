"""
Train an IT Incident Log Classifier.

Pipeline:
  1. Load CSV dataset
  2. TF-IDF vectorisation
  3. Train Logistic Regression (category) + Priority classifier
  4. Evaluate with classification report
  5. Export model artifacts to models/

Run: python src/train_model.py
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


def load_data(path: str = "data/it_incidents.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"📂 Loaded {len(df)} records from {path}")
    return df


def build_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=5.0,
                    class_weight="balanced",
                    solver="lbfgs",

                ),
            ),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame, target_col: str, label: str
) -> Pipeline:
    """Train a pipeline for a given target column and print metrics."""
    X = df["description"]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {label} Classifier  —  Accuracy: {acc:.2%}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred))

    return pipe, acc


def main():
    os.makedirs("models", exist_ok=True)

    df = load_data()

    # --- Category classifier ---
    cat_pipe, cat_acc = train_and_evaluate(
        df, target_col="category", label="Category"
    )
    joblib.dump(cat_pipe, "models/category_model.joblib")

    # --- Priority classifier ---
    pri_pipe, pri_acc = train_and_evaluate(
        df, target_col="priority", label="Priority"
    )
    joblib.dump(pri_pipe, "models/priority_model.joblib")

    # --- Team mapping ---
    team_map = df.drop_duplicates("category").set_index("category")["assigned_team"].to_dict()
    with open("models/team_map.json", "w") as f:
        json.dump(team_map, f, indent=2)

    # --- Save metrics for display in app ---
    metrics = {
        "category_accuracy": round(cat_acc, 4),
        "priority_accuracy": round(pri_acc, 4),
        "training_samples": len(df),
        "categories": sorted(df["category"].unique().tolist()),
        "priorities": sorted(df["priority"].unique().tolist()),
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Models and artifacts saved to models/")
    print(f"   • category_model.joblib")
    print(f"   • priority_model.joblib")
    print(f"   • team_map.json")
    print(f"   • metrics.json")


if __name__ == "__main__":
    main()
