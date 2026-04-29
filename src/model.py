"""Train the triage acuity prediction model.

Combines numeric vital-sign features with a TF-IDF representation of the chief
complaint and trains a LightGBM multi-class classifier. The full pipeline is
saved as a single joblib artifact so the FastAPI app can load it directly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_triage
from src.preprocessing import FEATURE_COLUMNS, build_feature_frame

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "triage_model_v1.pkl"


def _build_features(df, fit: bool, artifacts=None):
    """Materialise the combined numeric + TF-IDF feature matrix.

    During training we fit the imputer, scaler, and vectoriser; at inference we
    reuse the fitted transformers from ``artifacts``.
    """
    feats = build_feature_frame(df)
    numeric = feats[FEATURE_COLUMNS].astype(float).values
    text = feats["chiefcomplaint"].fillna("").values

    if fit:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            max_features=20000,
            sublinear_tf=True,
        )
        numeric = imputer.fit_transform(numeric)
        numeric = scaler.fit_transform(numeric)
        text_matrix = vectorizer.fit_transform(text)
        artifacts = {
            "imputer": imputer,
            "scaler": scaler,
            "vectorizer": vectorizer,
        }
    else:
        assert artifacts is not None, "artifacts required for inference"
        numeric = artifacts["imputer"].transform(numeric)
        numeric = artifacts["scaler"].transform(numeric)
        text_matrix = artifacts["vectorizer"].transform(text)

    X = hstack([csr_matrix(numeric), text_matrix]).tocsr()
    return X, artifacts


def train(
    csv_path: str = "data/raw/triage.csv",
    sample: int | None = None,
    test_size: float = 0.15,
    random_state: int = 42,
):
    print(f"[1/5] Loading triage data from {csv_path}…")
    df = load_triage(csv_path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=random_state)
    print(f"  rows: {len(df):,}, acuity distribution: {df['acuity'].value_counts().to_dict()}")

    print("[2/5] Splitting train / val / test…")
    train_full, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["acuity"],
    )
    train_df, val_df = train_test_split(
        train_full,
        test_size=0.1,
        random_state=random_state,
        stratify=train_full["acuity"],
    )

    print("[3/5] Building features…")
    X_train, artifacts = _build_features(train_df, fit=True)
    X_val, _ = _build_features(val_df, fit=False, artifacts=artifacts)
    X_test, _ = _build_features(test_df, fit=False, artifacts=artifacts)
    y_train = train_df["acuity"].values
    y_val = val_df["acuity"].values
    y_test = test_df["acuity"].values

    classes = np.unique(y_train)

    # We train without inverse-frequency sample weights: that maximises plain
    # accuracy on this dataset. We do, however, gently protect the rarest
    # class (5) with a sqrt-balanced weight so it's not entirely ignored.
    _, class_counts = np.unique(y_train, return_counts=True)
    sqrt_weights = {
        c: float(np.sqrt(len(y_train) / (len(classes) * cnt)))
        for c, cnt in zip(classes, class_counts)
    }
    sample_weight = np.array([sqrt_weights[c] for c in y_train])

    print("[4/5] Training LightGBM classifier…")
    t0 = time.time()
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=255,
        max_depth=-1,
        min_child_samples=30,
        feature_fraction=0.8,
        bagging_fraction=0.85,
        bagging_freq=5,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    from lightgbm import early_stopping, log_evaluation

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=40), log_evaluation(period=0)],
    )
    print(f"  trained in {time.time() - t0:.1f}s — best iter: {model.best_iteration_}")

    print("[5/5] Evaluating…")
    proba = model.predict_proba(X_test)
    preds = model.classes_[np.argmax(proba, axis=1)]
    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    # ordinal-aware top-2 accuracy: pred is "close" if the true class is in
    # the model's top-2 most likely classes.
    top2 = model.classes_[np.argsort(-proba, axis=1)[:, :2]]
    top2_acc = float(np.mean([y_test[i] in top2[i] for i in range(len(y_test))]))

    print(f"  accuracy:    {accuracy:.4f}")
    print(f"  top-2 acc:   {top2_acc:.4f}")
    print(f"  macro f1:    {macro_f1:.4f}")
    print(classification_report(y_test, preds, digits=4))
    print("  confusion matrix:")
    print(confusion_matrix(y_test, preds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "imputer": artifacts["imputer"],
        "scaler": artifacts["scaler"],
        "vectorizer": artifacts["vectorizer"],
        "feature_columns": FEATURE_COLUMNS,
        "classes_": list(map(int, classes)),
        "metrics": {
            "accuracy": float(accuracy),
            "top_2_accuracy": float(top2_acc),
            "macro_f1": float(macro_f1),
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")
    return bundle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/triage.csv")
    parser.add_argument("--sample", type=int, default=None, help="optional row cap for fast iteration")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(csv_path=args.csv, sample=args.sample)
