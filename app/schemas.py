"""Pydantic request/response schemas for the triage + platform APIs."""

from __future__ import annotations

from datetime import datetime
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


class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    mrn: Optional[str] = None


class PatientOut(BaseModel):
    id: int
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class IntakeCreate(BaseModel):
    patient: PatientCreate
    arrival_mode: str = "walk-in"
    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    pain: Optional[str] = None
    chief_complaint: str = ""
    notes: Optional[str] = None
    run_prediction: bool = True


class IntakeOut(BaseModel):
    id: int
    patient_id: int
    arrival_mode: str
    chief_complaint: str
    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    pain: Optional[str] = None
    predicted_acuity: Optional[int] = None
    acuity_label: Optional[str] = None
    confidence: Optional[float] = None
    disposition: str
    status: str
    notes: Optional[str] = None
    created_at: datetime
    patient: Optional[PatientOut] = None

    model_config = {"from_attributes": True}


class IntakeStatusUpdate(BaseModel):
    disposition: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class MetricOut(BaseModel):
    id: int
    metric_key: str
    metric_label: str
    value: float
    unit: str
    department: str
    recorded_at: datetime

    model_config = {"from_attributes": True}


class OpsDashboard(BaseModel):
    active_intakes: int
    waiting_count: int
    avg_acuity: Optional[float] = None
    acuity_distribution: Dict[str, int]
    metrics: List[MetricOut]
    recent_intakes: List[IntakeOut]


class AISummaryRequest(BaseModel):
    intake_id: int
    include_recommendations: bool = True


class AISummaryOut(BaseModel):
    id: int
    intake_id: int
    summary_text: str
    recommendations: str
    sources: str
    model_name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class WorkflowRecommendRequest(BaseModel):
    chief_complaint: str
    acuity: Optional[int] = None
    context: Optional[str] = None


class WorkflowRecommendOut(BaseModel):
    recommendations: str
    sources: List[str]
    model_name: str
