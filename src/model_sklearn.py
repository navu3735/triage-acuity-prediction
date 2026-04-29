"""Train a Vercel-safe triage model (no LightGBM / libgomp).

Vercel's Python runtime environment does not include `libgomp.so.1`, which
LightGBM (and some other boosted tree libraries) require for OpenMP. This
module trains a pure scikit-learn model instead:

  - Numeric vitals: impute (median) + standardize
  - Text: TF-IDF on chief complaint (1-2 grams)
  - Classifier: SGDClassifier (multinomial logistic regression via log-loss)

The resulting artifact is small, fast, and compatible with serverless.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_triage
from src.preprocessing import FEATURE_COLUMNS, build_feature_frame

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "triage_model_sklearn.pkl"


def _build_features(df, fit: bool, artifacts=None):
    feats = build_feature_frame(df)
    numeric = feats[FEATURE_COLUMNS].astype(float).values
    text = feats["chiefcomplaint"].fillna("").astype(str).values

    if fit:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            max_features=30000,
            sublinear_tf=True,
        )
        numeric = scaler.fit_transform(imputer.fit_transform(numeric))
        text_matrix = vectorizer.fit_transform(text)
        artifacts = {"imputer": imputer, "scaler": scaler, "vectorizer": vectorizer}
    else:
        numeric = artifacts["scaler"].transform(artifacts["imputer"].transform(numeric))
        text_matrix = artifacts["vectorizer"].transform(text)

    X = hstack([csr_matrix(numeric), text_matrix]).tocsr()
    return X, artifacts


def train(csv_path: str = "data/raw/triage.csv", sample: int | None = None):
    print(f"[1/5] Loading data from {csv_path} …")
    df = load_triage(csv_path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    print(f"  rows: {len(df):,}  acuity dist: {df['acuity'].value_counts().to_dict()}")

    print("[2/5] Split train/val/test …")
    train_full, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["acuity"])
    train_df, val_df = train_test_split(train_full, test_size=0.1, random_state=42, stratify=train_full["acuity"])

    print("[3/5] Feature extraction …")
    X_train, artifacts = _build_features(train_df, fit=True)
    X_val, _ = _build_features(val_df, fit=False, artifacts=artifacts)
    X_test, _ = _build_features(test_df, fit=False, artifacts=artifacts)
    y_train = train_df["acuity"].astype(int).values
    y_val = val_df["acuity"].astype(int).values
    y_test = test_df["acuity"].astype(int).values

    print("[4/5] Training SGDClassifier (log-loss) …")
    t0 = time.time()
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print(f"  trained in {time.time() - t0:.1f}s")

    print("[5/5] Evaluating …")
    proba = clf.predict_proba(X_test)
    preds = clf.classes_[np.argmax(proba, axis=1)]
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    top2 = clf.classes_[np.argsort(-proba, axis=1)[:, :2]]
    top2_acc = float(np.mean([y_test[i] in top2[i] for i in range(len(y_test))]))
    print(f"  accuracy:  {acc:.4f}")
    print(f"  top-2:     {top2_acc:.4f}")
    print(f"  macro f1:  {macro_f1:.4f}")
    print(classification_report(y_test, preds, digits=4))
    print("  confusion matrix:")
    print(confusion_matrix(y_test, preds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "imputer": artifacts["imputer"],
        "scaler": artifacts["scaler"],
        "vectorizer": artifacts["vectorizer"],
        "feature_columns": FEATURE_COLUMNS,
        "classes_": [int(c) for c in clf.classes_],
        "metrics": {"accuracy": float(acc), "top_2_accuracy": float(top2_acc), "macro_f1": float(macro_f1)},
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/raw/triage.csv")
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()
    train(csv_path=args.csv, sample=args.sample)

