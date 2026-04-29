"""Pydantic request/response schemas for the triage API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """User-supplied vital signs and free-text symptoms."""

    temperature: Optional[float] = Field(
        None, description="Body temperature in degrees Fahrenheit (e.g. 98.6)."
    )
    heartrate: Optional[float] = Field(None, description="Heart rate in beats per minute.")
    resprate: Optional[float] = Field(None, description="Respiratory rate in breaths per minute.")
    o2sat: Optional[float] = Field(None, description="Oxygen saturation in % (SpO2).")
    sbp: Optional[float] = Field(None, description="Systolic blood pressure in mmHg.")
    dbp: Optional[float] = Field(None, description="Diastolic blood pressure in mmHg.")
    pain: Optional[str] = Field(
        None,
        description="Pain rating 0-10 or descriptor (e.g. 'moderate', 'severe').",
    )
    chiefcomplaint: Optional[str] = Field(
        None,
        description="Free-text symptoms / chief complaint (e.g. 'chest pain, shortness of breath').",
    )


class TriageResponse(BaseModel):
    acuity: int
    label: str
    description: str
    confidence: float
    probabilities: Dict[str, float]
    top_features: List[str]
