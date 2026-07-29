"""ORM models for emergency intake and operational healthcare data."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    mrn: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    first_name: Mapped[str] = mapped_column(String(80))
    last_name: Mapped[str] = mapped_column(String(80))
    date_of_birth: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    intakes: Mapped[list["EmergencyIntake"]] = relationship(back_populates="patient")


class EmergencyIntake(Base):
    __tablename__ = "emergency_intakes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    arrival_mode: Mapped[str] = mapped_column(String(40), default="walk-in")
    chief_complaint: Mapped[str] = mapped_column(Text, default="")
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    heartrate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resprate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    o2sat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sbp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dbp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pain: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    predicted_acuity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    acuity_label: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    disposition: Mapped[str] = mapped_column(String(40), default="waiting")
    status: Mapped[str] = mapped_column(String(40), default="active")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patient: Mapped["Patient"] = relationship(back_populates="intakes")
    ai_summaries: Mapped[list["AISummary"]] = relationship(back_populates="intake")


class OperationalMetric(Base):
    __tablename__ = "operational_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    metric_key: Mapped[str] = mapped_column(String(64), index=True)
    metric_label: Mapped[str] = mapped_column(String(120))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32), default="")
    department: Mapped[str] = mapped_column(String(64), default="ED")
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AISummary(Base):
    __tablename__ = "ai_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    intake_id: Mapped[int] = mapped_column(ForeignKey("emergency_intakes.id"), index=True)
    summary_text: Mapped[str] = mapped_column(Text)
    recommendations: Mapped[str] = mapped_column(Text, default="")
    sources: Mapped[str] = mapped_column(Text, default="")
    model_name: Mapped[str] = mapped_column(String(80), default="langchain-rag")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    intake: Mapped["EmergencyIntake"] = relationship(back_populates="ai_summaries")
