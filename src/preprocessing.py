"""Feature engineering utilities for triage data.

Cleans vital signs, normalizes the free-text pain column, and exposes a
``build_feature_frame`` helper that is used by both training and inference so
the two stay in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

NUMERIC_VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]


VITAL_BOUNDS = {
    "temperature": (80.0, 110.0),
    "heartrate": (20.0, 250.0),
    "resprate": (4.0, 60.0),
    "o2sat": (40.0, 100.0),
    "sbp": (40.0, 260.0),
    "dbp": (20.0, 200.0),
}


PAIN_TEXT_TO_SCORE = {
    "none": 0,
    "no": 0,
    "0": 0,
    "mild": 2,
    "low": 2,
    "minimal": 1,
    "moderate": 5,
    "med": 5,
    "medium": 5,
    "high": 8,
    "severe": 9,
    "extreme": 10,
    "worst": 10,
    "critical": 10,
    "10/10": 10,
}


def _coerce_pain(value) -> float:
    """Convert the messy ``pain`` column into a 0-10 numeric score.

    The dataset stores pain as a string with a mix of numbers ("7"), ranges
    ("2-3"), descriptors ("moderate", "critical"), and "unable to assess"
    placeholders. We map known text values to a numeric score, parse ranges by
    averaging, and fall back to NaN for unknown / unable values so downstream
    imputation can take over.
    """

    if value is None:
        return np.nan
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(np.clip(value, 0, 10))
    text = str(value).strip().lower()
    if not text or text in {"nan", "uta", "unable", "unable to assess", "non-verbal", "ett"}:
        return np.nan
    if text.replace(".", "", 1).isdigit():
        return float(np.clip(float(text), 0, 10))
    if "-" in text:
        parts = text.split("-")
        try:
            nums = [float(p) for p in parts if p.replace(".", "", 1).isdigit()]
            if nums:
                return float(np.clip(sum(nums) / len(nums), 0, 10))
        except ValueError:
            pass
    if text in PAIN_TEXT_TO_SCORE:
        return float(PAIN_TEXT_TO_SCORE[text])
    for key, score in PAIN_TEXT_TO_SCORE.items():
        if key in text:
            return float(score)
    return np.nan


def _clip_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Mask physiologically impossible vital sign values as NaN."""
    out = df.copy()
    for col, (lo, hi) in VITAL_BOUNDS.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan
    return out


@dataclass
class TriageRecord:
    """Schema for a single inference request."""

    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    pain: Optional[str] = None
    chiefcomplaint: Optional[str] = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing logic at train and inference time."""
    out = _clip_vitals(df)
    if "pain" in out.columns:
        out["pain"] = out["pain"].apply(_coerce_pain)
    if "chiefcomplaint" in out.columns:
        out["chiefcomplaint"] = (
            out["chiefcomplaint"].fillna("").astype(str).str.lower().str.strip()
        )
    else:
        out["chiefcomplaint"] = ""

    if {"sbp", "dbp"}.issubset(out.columns):
        out["pulse_pressure"] = out["sbp"] - out["dbp"]
        out["map"] = out["dbp"] + (out["sbp"] - out["dbp"]) / 3.0
    if {"heartrate", "sbp"}.issubset(out.columns):
        out["shock_index"] = out["heartrate"] / out["sbp"].replace(0, np.nan)

    return out


FEATURE_COLUMNS = NUMERIC_VITALS + ["pain", "pulse_pressure", "map", "shock_index"]
