"""FastAPI app exposing the triage acuity prediction model.

Loads the Keras deep-learning model (text-embedding + numeric branches) by
default, and falls back to the LightGBM bundle if the DL artifacts are
missing. The browser frontend is rendered from ``templates/index.html``;
``/predict`` accepts JSON describing the patient and returns the predicted
ESI level along with the full probability distribution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# Quiet TF logs before any tensorflow import lower in this module.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.schemas import TriageRequest, TriageResponse
from src.preprocessing import FEATURE_COLUMNS, build_feature_frame

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DL_MODEL_PATH = ROOT_DIR / "models" / "triage_model_dl.keras"
DL_ARTIFACTS_PATH = ROOT_DIR / "models" / "triage_dl_artifacts.pkl"
LGBM_MODEL_PATH = ROOT_DIR / "models" / "triage_model_v1.pkl"


ACUITY_LABELS = {
    1: ("Resuscitation", "Immediately life-threatening — needs lifesaving intervention now."),
    2: ("Emergent", "High risk situation — should be seen within ~10 minutes."),
    3: ("Urgent", "Stable but needs multiple resources — typical wait up to 30-60 minutes."),
    4: ("Less Urgent", "One resource expected — non-urgent but still requires care."),
    5: ("Non-Urgent", "Minor complaint — no resources expected, longest wait acceptable."),
}

# When the user leaves a vital sign blank we substitute these healthy-adult
# normal values *before* the imputer runs. The imputer was trained on the
# median of an ED population (which skews toward sicker patients), so without
# this override empty fields would silently push the prediction toward higher
# acuity classes.
DEFAULT_NORMAL_VITALS: Dict[str, object] = {
    "temperature": 98.6,
    "heartrate": 75,
    "resprate": 14,
    "o2sat": 99,
    "sbp": 120,
    "dbp": 78,
    "pain": "0",
}

# Fallback class counts used if the loaded artifact bundle is older than the
# prior-calibration upgrade (the artifacts pickle has since been patched, but
# we keep these for safety).
FALLBACK_NATURAL_COUNTS = {1: 18374, 2: 106649, 3: 172175, 4: 21806, 5: 842}
FALLBACK_BALANCED_COUNTS = {1: 25000, 2: 106649, 3: 172175, 4: 25000, 5: 25000}


def _tensorflow_is_installed() -> bool:
    """Return True iff TensorFlow is importable (serverless images often omit it to save ~2 GB).

    Checking without importing heavyweight submodules aggressively is enough for our branch.
    """

    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


app = FastAPI(
    title="Triage Acuity Predictor",
    description="Predicts Emergency Severity Index (ESI 1-5) from vital signs and symptoms.",
    version="2.0.0",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Cached predictor (lazy-initialised at first request / startup).
_predictor: "Predictor | None" = None


class Predictor:
    """Wraps either the Keras DL model or the LightGBM fallback behind one
    interface.  Keeps the FastAPI handler agnostic of which model is loaded.
    """

    def __init__(self):
        self.kind: str = ""
        self.classes_: List[int] = []
        self.metrics: Dict[str, float] = {}
        self._init()

    # -- model loading --------------------------------------------------- #

    def _init(self):
        dl_ready = (
            DL_MODEL_PATH.exists()
            and DL_ARTIFACTS_PATH.exists()
            and _tensorflow_is_installed()
        )
        if dl_ready:
            try:
                self._init_dl()
                return
            except Exception:
                pass
        if LGBM_MODEL_PATH.exists():
            self._init_lgbm()
        else:
            raise RuntimeError(
                "No runnable model found. For production (e.g. Vercel without TensorFlow) "
                "commit `models/triage_model_v1.pkl`. Locally, train via "
                "`python -m src.model`, or DL via `pip install -r requirements-extra-dl.txt` "
                "then `python -m src.train_dl`."
            )

    def _init_dl(self):
        import tensorflow as tf

        artifacts = joblib.load(DL_ARTIFACTS_PATH)
        self.tokenizer = artifacts["tokenizer"]
        self.imputer = artifacts["imputer"]
        self.scaler = artifacts["scaler"]
        self.classes_ = artifacts["classes_"]
        self.seq_len = artifacts["seq_len"]
        self.metrics = artifacts.get("metrics", {})
        self.model = tf.keras.models.load_model(DL_MODEL_PATH)
        self.kind = "deep-learning"
        self._init_calibration(
            artifacts.get("natural_class_counts", FALLBACK_NATURAL_COUNTS),
            artifacts.get("balanced_class_counts", FALLBACK_BALANCED_COUNTS),
        )

    def _init_lgbm(self):
        bundle = joblib.load(LGBM_MODEL_PATH)
        self.imputer = bundle["imputer"]
        self.scaler = bundle["scaler"]
        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]
        self.classes_ = bundle["classes_"]
        self.metrics = bundle.get("metrics", {})
        self.kind = "lightgbm"
        # The LightGBM model used sample_weight, not oversampling, so its
        # output is already on the natural distribution — calibration is a no-op.
        self.calibration = np.ones(len(self.classes_), dtype="float32")

    def _init_calibration(self, natural_counts: Dict[int, int], balanced_counts: Dict[int, int]):
        """Build a per-class adjustment vector that converts probabilities
        from the *training* (oversampled) distribution back to the natural
        prior. This is the standard prior-correction trick: when training on
        a re-sampled distribution, multiply predictions by
        ``true_prior / training_prior`` and renormalise.
        """
        n_total = float(sum(natural_counts.values()))
        b_total = float(sum(balanced_counts.values()))
        self.natural_priors = {c: natural_counts[c] / n_total for c in self.classes_}
        self.balanced_priors = {c: balanced_counts[c] / b_total for c in self.classes_}
        self.calibration = np.array(
            [self.natural_priors[c] / self.balanced_priors[c] for c in self.classes_],
            dtype="float32",
        )

    def _calibrate(self, proba: np.ndarray) -> np.ndarray:
        adjusted = proba * self.calibration
        s = adjusted.sum()
        return adjusted / s if s > 0 else proba

    # -- prediction ------------------------------------------------------ #

    def predict_proba(self, payload: TriageRequest) -> np.ndarray:
        import pandas as pd

        data = payload.model_dump()
        # Substitute healthy-adult defaults for any vital sign the user didn't
        # supply. Without this, the imputer fills in the median of an ED
        # population and the model treats unknown vitals as "abnormal".
        for k, v in DEFAULT_NORMAL_VITALS.items():
            if data.get(k) in (None, ""):
                data[k] = v

        df = pd.DataFrame([data])
        feats = build_feature_frame(df)
        numeric = feats[FEATURE_COLUMNS].astype(float).values
        text = feats["chiefcomplaint"].fillna("").astype(str).values

        numeric = self.imputer.transform(numeric)
        numeric = self.scaler.transform(numeric).astype("float32")

        if self.kind == "deep-learning":
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            seqs = self.tokenizer.texts_to_sequences(text)
            seqs = pad_sequences(
                seqs, maxlen=self.seq_len, padding="post", truncating="post"
            ).astype("int32")
            proba = self.model.predict(
                {"text": seqs, "numeric": numeric}, verbose=0
            )[0]
        else:
            # LightGBM path
            from scipy.sparse import csr_matrix, hstack

            text_matrix = self.vectorizer.transform(text)
            X = hstack([csr_matrix(numeric), text_matrix]).tocsr()
            proba = self.model.predict_proba(X)[0]

        return self._calibrate(proba)

    def top_features(self, payload: TriageRequest, n: int = 5) -> List[str]:
        text = (payload.chiefcomplaint or "").lower().strip()
        if not text:
            return []
        if self.kind == "deep-learning":
            # No TF-IDF weights to rank with — return the in-vocab tokens we
            # actually fed to the model so the user sees what was recognised.
            seq = self.tokenizer.texts_to_sequences([text])[0][: self.seq_len]
            index_word = self.tokenizer.index_word
            tokens = []
            seen = set()
            for idx in seq:
                w = index_word.get(idx)
                if w and w != "<OOV>" and w not in seen:
                    tokens.append(w)
                    seen.add(w)
                if len(tokens) >= n:
                    break
            return tokens
        # LightGBM path: highest-weighted TF-IDF tokens that matched.
        vec = self.vectorizer.transform([text])
        if vec.nnz == 0:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        arr = vec.toarray().ravel()
        idx = np.argsort(arr)[::-1][:n]
        return [feature_names[i] for i in idx if arr[i] > 0]


def _get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


@app.on_event("startup")
def _startup():
    try:
        p = _get_predictor()
        print(f"[startup] loaded {p.kind} model • metrics={p.metrics}")
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] model not yet available: {exc}")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> Dict[str, object]:
    try:
        p = _get_predictor()
        return {
            "status": "ok",
            "model_kind": p.kind,
            "metrics": p.metrics,
            "classes": p.classes_,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "detail": str(exc)}


@app.post("/predict", response_model=TriageResponse)
def predict(payload: TriageRequest) -> TriageResponse:
    if all(
        getattr(payload, f) in (None, "")
        for f in [
            "temperature", "heartrate", "resprate", "o2sat",
            "sbp", "dbp", "pain", "chiefcomplaint",
        ]
    ):
        raise HTTPException(status_code=400, detail="At least one field is required.")

    try:
        p = _get_predictor()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    probas = p.predict_proba(payload)
    pred_idx = int(np.argmax(probas))
    pred_class = int(p.classes_[pred_idx])
    label, description = ACUITY_LABELS.get(pred_class, ("Unknown", ""))
    proba_map = {str(int(c)): float(round(prob, 4)) for c, prob in zip(p.classes_, probas)}

    return TriageResponse(
        acuity=pred_class,
        label=label,
        description=description,
        confidence=float(round(probas[pred_idx], 4)),
        probabilities=proba_map,
        top_features=p.top_features(payload),
    )
