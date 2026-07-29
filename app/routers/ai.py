"""LangChain RAG endpoints for summaries and workflow recommendations."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.db import get_db
from app.models import AISummary, EmergencyIntake
from app.rag.pipelines import run_patient_summary_chain, run_workflow_recommendation_chain
from app.schemas import (
    AISummaryOut,
    AISummaryRequest,
    WorkflowRecommendOut,
    WorkflowRecommendRequest,
)

router = APIRouter(prefix="/api/ai", tags=["ai-rag"])


@router.post("/summarize", response_model=AISummaryOut)
def summarize_intake(body: AISummaryRequest, db: Session = Depends(get_db)):
    intake = (
        db.query(EmergencyIntake)
        .options(joinedload(EmergencyIntake.patient))
        .filter(EmergencyIntake.id == body.intake_id)
        .first()
    )
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")

    patient = intake.patient
    patient_blob = (
        f"Patient {patient.first_name} {patient.last_name} (MRN {patient.mrn}). "
        f"Arrival: {intake.arrival_mode}. "
        f"Chief complaint: {intake.chief_complaint}. "
        f"Vitals: temp={intake.temperature}, HR={intake.heartrate}, RR={intake.resprate}, "
        f"SpO2={intake.o2sat}, BP={intake.sbp}/{intake.dbp}, pain={intake.pain}. "
        f"Predicted ESI: {intake.predicted_acuity} ({intake.acuity_label}), "
        f"confidence={intake.confidence}."
    )

    try:
        summary, recs, sources, model_name = run_patient_summary_chain(
            patient_blob,
            acuity=intake.predicted_acuity,
            include_recommendations=body.include_recommendations,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {exc}") from exc

    row = AISummary(
        intake_id=intake.id,
        summary_text=summary,
        recommendations=recs,
        sources="\n".join(sources),
        model_name=model_name,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/summaries/{intake_id}", response_model=List[AISummaryOut])
def list_summaries(intake_id: int, db: Session = Depends(get_db)):
    return (
        db.query(AISummary)
        .filter(AISummary.intake_id == intake_id)
        .order_by(AISummary.created_at.desc())
        .all()
    )


@router.post("/recommend", response_model=WorkflowRecommendOut)
def recommend_workflow(body: WorkflowRecommendRequest):
    try:
        text, sources, model_name = run_workflow_recommendation_chain(
            chief_complaint=body.chief_complaint,
            acuity=body.acuity,
            context=body.context,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {exc}") from exc
    return WorkflowRecommendOut(
        recommendations=text,
        sources=sources,
        model_name=model_name,
    )
