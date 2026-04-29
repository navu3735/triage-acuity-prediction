"""Lightweight data loading helpers for the MIMIC-IV-ED triage csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TRIAGE_CSV = Path("data/raw/triage.csv")


def load_triage(path: Path | str = DEFAULT_TRIAGE_CSV) -> pd.DataFrame:
    """Load the raw triage table and drop rows without an acuity label."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["acuity"]).copy()
    df["acuity"] = df["acuity"].astype(int)
    return df
