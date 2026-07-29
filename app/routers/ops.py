"""Operational healthcare data APIs."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.db import get_db
from app.models import EmergencyIntake, OperationalMetric
from app.schemas import IntakeOut, MetricOut, OpsDashboard

router = APIRouter(prefix="/api/ops", tags=["operations"])


@router.get("/metrics", response_model=List[MetricOut])
def list_metrics(db: Session = Depends(get_db)):
    return (
        db.query(OperationalMetric)
        .order_by(OperationalMetric.recorded_at.desc())
        .limit(50)
        .all()
    )


@router.get("/dashboard", response_model=OpsDashboard)
def dashboard(db: Session = Depends(get_db)):
    active = (
        db.query(EmergencyIntake)
        .filter(EmergencyIntake.status == "active")
        .count()
    )
    waiting = (
        db.query(EmergencyIntake)
        .filter(
            EmergencyIntake.status == "active",
            EmergencyIntake.disposition == "waiting",
        )
        .count()
    )
    avg_acuity = (
        db.query(func.avg(EmergencyIntake.predicted_acuity))
        .filter(EmergencyIntake.predicted_acuity.isnot(None))
        .scalar()
    )
    dist_rows = (
        db.query(EmergencyIntake.predicted_acuity, func.count())
        .filter(EmergencyIntake.predicted_acuity.isnot(None))
        .group_by(EmergencyIntake.predicted_acuity)
        .all()
    )
    acuity_distribution = {str(k): int(v) for k, v in dist_rows if k is not None}
    metrics = (
        db.query(OperationalMetric)
        .order_by(OperationalMetric.recorded_at.desc())
        .limit(12)
        .all()
    )
    recent = (
        db.query(EmergencyIntake)
        .options(joinedload(EmergencyIntake.patient))
        .order_by(EmergencyIntake.created_at.desc())
        .limit(8)
        .all()
    )
    return OpsDashboard(
        active_intakes=active,
        waiting_count=waiting,
        avg_acuity=float(round(avg_acuity, 2)) if avg_acuity is not None else None,
        acuity_distribution=acuity_distribution,
        metrics=metrics,
        recent_intakes=recent,
    )
