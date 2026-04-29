"""Deep-learning triage classifier (Keras / TensorFlow).

A multi-input neural network:

  * Text branch  -> Embedding -> Conv1D -> GlobalMaxPool
  * Numeric branch -> Dense -> BatchNorm -> Dropout
  * Concatenate -> Dense head -> softmax over ESI 1-5

Training data is balanced via *random oversampling* of the rare classes
(especially ESI 5, which is < 0.3% of the dataset). The held-out test set is
left at the natural distribution so reported accuracy is honest.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Keep TF a little quieter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.data_loader import load_triage
from src.preprocessing import FEATURE_COLUMNS, build_feature_frame

MODEL_DIR = Path("models")
KERAS_MODEL_PATH = MODEL_DIR / "triage_model_dl.keras"
ARTIFACTS_PATH = MODEL_DIR / "triage_dl_artifacts.pkl"
METRICS_PATH = MODEL_DIR / "triage_dl_metrics.json"

VOCAB_SIZE = 20000
SEQ_LEN = 32
EMBED_DIM = 96
LSTM_UNITS = 64
# Minimum number of training samples we want for *every* class. Classes
# already larger than this are kept at their original size — we never throw
# information away from the majority classes. Smaller classes are upsampled
# with replacement to reach ``MIN_PER_CLASS``. Smaller value means less
# duplication noise on the rare ESI 5 class.
MIN_PER_CLASS = 25000
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #

def _balance_via_oversampling(df: pd.DataFrame, min_per_class: int) -> pd.DataFrame:
    """Lift every class to at least ``min_per_class`` rows by oversampling.

    Classes already larger than the floor are kept at their original size —
    we never downsample majority classes (that throws away information and
    hurts overall accuracy). Only used on the *training* fold; validation
    and test stay at the natural distribution so reported accuracy is honest.
    """
    pieces = []
    for cls, sub in df.groupby("acuity"):
        n = len(sub)
        if n >= min_per_class:
            sampled = sub
            tag = "kept"
        else:
            sampled = resample(
                sub, replace=True, n_samples=min_per_class, random_state=RANDOM_STATE,
            )
            tag = f"oversampled {min_per_class / n:0.1f}x"
        print(f"  class {cls}: {n:>7,} -> {len(sampled):>7,}  ({tag})")
        pieces.append(sampled)
    out = pd.concat(pieces, axis=0).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out


def _prepare_inputs(df: pd.DataFrame, tokenizer, imputer, scaler, fit: bool):
    """Convert a dataframe into the (numeric, text-sequence) tuple the model expects."""
    feats = build_feature_frame(df)
    numeric = feats[FEATURE_COLUMNS].astype(float).values
    text = feats["chiefcomplaint"].fillna("").astype(str).values

    if fit:
        imputer.fit(numeric)
        numeric = imputer.transform(numeric)
        scaler.fit(numeric)
        numeric = scaler.transform(numeric)
        tokenizer.fit_on_texts(text)
    else:
        numeric = imputer.transform(numeric)
        numeric = scaler.transform(numeric)

    seqs = tokenizer.texts_to_sequences(text)
    seqs = pad_sequences(seqs, maxlen=SEQ_LEN, padding="post", truncating="post")
    return numeric.astype("float32"), seqs.astype("int32")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(num_numeric_features: int, num_classes: int) -> tf.keras.Model:
    # ----- text branch: Embedding -> multi-kernel Conv1D -> BiLSTM -> Pool -----
    text_in = layers.Input(shape=(SEQ_LEN,), name="text", dtype="int32")
    x_text = layers.Embedding(
        input_dim=VOCAB_SIZE + 1,
        output_dim=EMBED_DIM,
        mask_zero=False,  # disable masking so LSTM/Conv can run efficiently
        name="embedding",
    )(text_in)
    x_text = layers.SpatialDropout1D(0.25)(x_text)
    conv_a = layers.Conv1D(96, 2, activation="relu", padding="same")(x_text)
    conv_b = layers.Conv1D(96, 3, activation="relu", padding="same")(x_text)
    conv_c = layers.Conv1D(96, 5, activation="relu", padding="same")(x_text)
    x_text = layers.Concatenate()([conv_a, conv_b, conv_c])
    x_text = layers.SpatialDropout1D(0.2)(x_text)
    # BiLSTM lets the model capture symptom *combinations*
    # ("chest pain" + "shortness of breath" + "diaphoresis" -> ESI 1).
    x_text = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True, recurrent_dropout=0.0),
        name="bilstm",
    )(x_text)
    pool_max = layers.GlobalMaxPooling1D()(x_text)
    pool_avg = layers.GlobalAveragePooling1D()(x_text)
    x_text = layers.Concatenate()([pool_max, pool_avg])
    x_text = layers.Dropout(0.3)(x_text)
    x_text = layers.Dense(96, activation="relu")(x_text)

    # ----- numeric branch: Dense + BN + Dropout -----
    num_in = layers.Input(shape=(num_numeric_features,), name="numeric")
    x_num = layers.Dense(64, activation="relu")(num_in)
    x_num = layers.BatchNormalization()(x_num)
    x_num = layers.Dropout(0.2)(x_num)
    x_num = layers.Dense(32, activation="relu")(x_num)

    # ----- fusion + classifier head -----
    merged = layers.Concatenate()([x_text, x_num])
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="acuity")(x)

    model = tf.keras.Model(inputs=[text_in, num_in], outputs=out, name="triage_dl")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )
    return model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(csv_path: str = "data/raw/triage.csv", epochs: int = 20, batch_size: int = 512):
    print(f"[1/6] Loading data from {csv_path} ...")
    df = load_triage(csv_path)
    print(f"  rows: {len(df):,}  acuity dist: {df['acuity'].value_counts().to_dict()}")

    print("[2/6] Train/val/test split (stratified, natural distribution preserved on val+test) ...")
    train_full, test_df = train_test_split(
        df, test_size=0.15, random_state=RANDOM_STATE, stratify=df["acuity"]
    )
    train_df, val_df = train_test_split(
        train_full, test_size=0.1, random_state=RANDOM_STATE, stratify=train_full["acuity"]
    )

    natural_counts = train_df["acuity"].value_counts().to_dict()
    print(f"[3/6] Lifting minority classes to >= {MIN_PER_CLASS:,} (no downsampling) ...")
    train_df = _balance_via_oversampling(train_df, min_per_class=MIN_PER_CLASS)
    balanced_counts = train_df["acuity"].value_counts().to_dict()
    print(f"  balanced train rows: {len(train_df):,}")

    print("[4/6] Tokenizer + scaler + imputer ...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", lower=True)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_num_tr, X_txt_tr = _prepare_inputs(train_df, tokenizer, imputer, scaler, fit=True)
    X_num_val, X_txt_val = _prepare_inputs(val_df, tokenizer, imputer, scaler, fit=False)
    X_num_te, X_txt_te = _prepare_inputs(test_df, tokenizer, imputer, scaler, fit=False)

    classes = sorted(df["acuity"].unique().tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_tr = np.array([cls_to_idx[c] for c in train_df["acuity"].values])
    y_val = np.array([cls_to_idx[c] for c in val_df["acuity"].values])
    y_te = np.array([cls_to_idx[c] for c in test_df["acuity"].values])

    print(f"  vocab size in use: {min(VOCAB_SIZE, len(tokenizer.word_index)):,}")
    print(f"  numeric feature dim: {X_num_tr.shape[1]}")

    print("[5/6] Building & training model ...")
    model = build_model(num_numeric_features=X_num_tr.shape[1], num_classes=len(classes))
    model.summary(print_fn=lambda s: print("  " + s))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    t0 = time.time()
    history = model.fit(
        x={"text": X_txt_tr, "numeric": X_num_tr},
        y=y_tr,
        validation_data=({"text": X_txt_val, "numeric": X_num_val}, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    print(f"  trained in {time.time() - t0:.1f}s")

    print("[6/6] Evaluating on held-out test set (natural distribution) ...")
    proba = model.predict({"text": X_txt_te, "numeric": X_num_te}, batch_size=batch_size, verbose=0)
    preds = np.argmax(proba, axis=1)
    accuracy = accuracy_score(y_te, preds)
    macro_f1 = f1_score(y_te, preds, average="macro")
    top2 = np.argsort(-proba, axis=1)[:, :2]
    top2_acc = float(np.mean([y_te[i] in top2[i] for i in range(len(y_te))]))

    print(f"  accuracy:    {accuracy:.4f}")
    print(f"  top-2 acc:   {top2_acc:.4f}")
    print(f"  macro f1:    {macro_f1:.4f}")
    pred_labels = [classes[i] for i in preds]
    true_labels = [classes[i] for i in y_te]
    print(classification_report(true_labels, pred_labels, digits=4))
    print("  confusion matrix:")
    print(confusion_matrix(true_labels, pred_labels))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(KERAS_MODEL_PATH)
    joblib.dump(
        {
            "tokenizer": tokenizer,
            "imputer": imputer,
            "scaler": scaler,
            "classes_": classes,
            "feature_columns": FEATURE_COLUMNS,
            "seq_len": SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "natural_class_counts": {int(k): int(v) for k, v in natural_counts.items()},
            "balanced_class_counts": {int(k): int(v) for k, v in balanced_counts.items()},
            "metrics": {
                "accuracy": float(accuracy),
                "top_2_accuracy": float(top2_acc),
                "macro_f1": float(macro_f1),
            },
        },
        ARTIFACTS_PATH,
    )
    METRICS_PATH.write_text(
        json.dumps(
            {
                "accuracy": float(accuracy),
                "top_2_accuracy": float(top2_acc),
                "macro_f1": float(macro_f1),
                "history": {k: [float(x) for x in v] for k, v in history.history.items()},
            },
            indent=2,
        )
    )
    print(f"Saved Keras model -> {KERAS_MODEL_PATH}")
    print(f"Saved artifacts   -> {ARTIFACTS_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/triage.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    train(csv_path=args.csv, epochs=args.epochs, batch_size=args.batch)
