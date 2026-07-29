"""Emergency intake REST API."""

from __future__ import annotations

import secrets
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.db import get_db
from app.models import EmergencyIntake, Patient
from app.schemas import IntakeCreate, IntakeOut, IntakeStatusUpdate, PatientOut, TriageRequest

router = APIRouter(prefix="/api/intake", tags=["emergency-intake"])


def _mrn() -> str:
    return f"SF-{secrets.token_hex(3).upper()}"


def _run_prediction(payload: TriageRequest):
    from app.main import _get_predictor, ACUITY_LABELS
    import numpy as np

    p = _get_predictor()
    probas = p.predict_proba(payload)
    pred_idx = int(np.argmax(probas))
    pred_class = int(p.classes_[pred_idx])
    label, _ = ACUITY_LABELS.get(pred_class, ("Unknown", ""))
    return pred_class, label, float(round(probas[pred_idx], 4))


@router.post("", response_model=IntakeOut)
def create_intake(body: IntakeCreate, db: Session = Depends(get_db)):
    patient_data = body.patient
    mrn = patient_data.mrn or _mrn()
    existing = db.query(Patient).filter(Patient.mrn == mrn).first()
    if existing:
        patient = existing
    else:
        patient = Patient(
            mrn=mrn,
            first_name=patient_data.first_name,
            last_name=patient_data.last_name,
            date_of_birth=patient_data.date_of_birth,
            sex=patient_data.sex,
        )
        db.add(patient)
        db.flush()

    acuity = label = confidence = None
    if body.run_prediction:
        try:
            triage = TriageRequest(
                temperature=body.temperature,
                heartrate=body.heartrate,
                resprate=body.resprate,
                o2sat=body.o2sat,
                sbp=body.sbp,
                dbp=body.dbp,
                pain=body.pain,
                chiefcomplaint=body.chief_complaint,
            )
            acuity, label, confidence = _run_prediction(triage)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Prediction unavailable: {exc}") from exc

    intake = EmergencyIntake(
        patient_id=patient.id,
        arrival_mode=body.arrival_mode,
        chief_complaint=body.chief_complaint,
        temperature=body.temperature,
        heartrate=body.heartrate,
        resprate=body.resprate,
        o2sat=body.o2sat,
        sbp=body.sbp,
        dbp=body.dbp,
        pain=body.pain,
        predicted_acuity=acuity,
        acuity_label=label,
        confidence=confidence,
        notes=body.notes,
        disposition="waiting",
        status="active",
    )
    db.add(intake)
    db.commit()
    db.refresh(intake)
    intake.patient = patient
    return intake


@router.get("", response_model=List[IntakeOut])
def list_intakes(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    q = db.query(EmergencyIntake).options(joinedload(EmergencyIntake.patient))
    if status:
        q = q.filter(EmergencyIntake.status == status)
    rows = q.order_by(EmergencyIntake.created_at.desc()).limit(min(limit, 200)).all()
    return rows


@router.get("/patients/all", response_model=List[PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).order_by(Patient.created_at.desc()).all()


@router.get("/{intake_id}", response_model=IntakeOut)
def get_intake(intake_id: int, db: Session = Depends(get_db)):
    row = (
        db.query(EmergencyIntake)
        .options(joinedload(EmergencyIntake.patient))
        .filter(EmergencyIntake.id == intake_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Intake not found")
    return row


@router.patch("/{intake_id}", response_model=IntakeOut)
def update_intake(intake_id: int, body: IntakeStatusUpdate, db: Session = Depends(get_db)):
    row = (
        db.query(EmergencyIntake)
        .options(joinedload(EmergencyIntake.patient))
        .filter(EmergencyIntake.id == intake_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Intake not found")
    if body.disposition is not None:
        row.disposition = body.disposition
    if body.status is not None:
        row.status = body.status
    if body.notes is not None:
        row.notes = body.notes
    db.commit()
    db.refresh(row)
    return row
