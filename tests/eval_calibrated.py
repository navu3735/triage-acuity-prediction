"""Re-evaluate the saved Keras model on the natural test fold WITH the
prior calibration applied. Confirms calibration doesn't sacrifice accuracy.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.data_loader import load_triage
from src.preprocessing import FEATURE_COLUMNS, build_feature_frame

MODEL = tf.keras.models.load_model("models/triage_model_dl.keras")
ART = joblib.load("models/triage_dl_artifacts.pkl")

# Reproduce the same split as training (random_state=42)
df = load_triage("data/raw/triage.csv")
train_full, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["acuity"])

feats = build_feature_frame(test_df)
numeric = feats[FEATURE_COLUMNS].astype(float).values
text = feats["chiefcomplaint"].fillna("").astype(str).values
numeric = ART["scaler"].transform(ART["imputer"].transform(numeric)).astype("float32")
seqs = pad_sequences(
    ART["tokenizer"].texts_to_sequences(text),
    maxlen=ART["seq_len"], padding="post", truncating="post",
).astype("int32")

probas = MODEL.predict({"text": seqs, "numeric": numeric}, verbose=0, batch_size=512)

# build calibration factors
nat = ART["natural_class_counts"]
bal = ART["balanced_class_counts"]
classes = ART["classes_"]
n_total = sum(nat.values())
b_total = sum(bal.values())
calib = np.array([(nat[c] / n_total) / (bal[c] / b_total) for c in classes])
print("calibration factors:", dict(zip(classes, [f"{x:.3f}" for x in calib])))

y_true = test_df["acuity"].values
cls_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([cls_to_idx[c] for c in y_true])

for label, p in [("uncalibrated", probas), ("calibrated", probas * calib / (probas * calib).sum(axis=1, keepdims=True))]:
    pred = np.argmax(p, axis=1)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    top2 = np.argsort(-p, axis=1)[:, :2]
    top2_acc = float(np.mean([y[i] in top2[i] for i in range(len(y))]))
    print(f"\n=== {label} ===")
    print(f"accuracy={acc:.4f}  top2={top2_acc:.4f}  macro_f1={f1:.4f}")
    pred_labels = [classes[i] for i in pred]
    true_labels = [classes[i] for i in y]
    print(classification_report(true_labels, pred_labels, digits=3))
    print(confusion_matrix(true_labels, pred_labels))
