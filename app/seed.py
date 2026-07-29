"""Seed sample operational metrics and demo patients."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models import EmergencyIntake, OperationalMetric, Patient


def seed_if_empty(db: Session) -> None:
    if db.query(OperationalMetric).count() == 0:
        now = datetime.utcnow()
        metrics = [
            OperationalMetric(
                metric_key="ed_census",
                metric_label="ED Census",
                value=42,
                unit="patients",
                department="ED",
                recorded_at=now,
            ),
            OperationalMetric(
                metric_key="avg_door_to_doc",
                metric_label="Avg Door-to-Doc",
                value=18,
                unit="min",
                department="ED",
                recorded_at=now,
            ),
            OperationalMetric(
                metric_key="bed_occupancy",
                metric_label="Bed Occupancy",
                value=87,
                unit="%",
                department="ED",
                recorded_at=now,
            ),
            OperationalMetric(
                metric_key="lwbs_rate",
                metric_label="Left Without Being Seen",
                value=2.4,
                unit="%",
                department="ED",
                recorded_at=now,
            ),
            OperationalMetric(
                metric_key="icu_available",
                metric_label="ICU Beds Available",
                value=3,
                unit="beds",
                department="ICU",
                recorded_at=now,
            ),
            OperationalMetric(
                metric_key="lab_turnaround",
                metric_label="Lab Turnaround (STAT)",
                value=24,
                unit="min",
                department="Lab",
                recorded_at=now,
            ),
        ]
        db.add_all(metrics)

    if db.query(Patient).count() == 0:
        patients = [
            Patient(
                mrn="SF-1001",
                first_name="Maya",
                last_name="Chen",
                date_of_birth="1988-04-12",
                sex="F",
            ),
            Patient(
                mrn="SF-1002",
                first_name="Jordan",
                last_name="Okeke",
                date_of_birth="1971-11-03",
                sex="M",
            ),
            Patient(
                mrn="SF-1003",
                first_name="Elena",
                last_name="Vasquez",
                date_of_birth="1995-07-22",
                sex="F",
            ),
        ]
        db.add_all(patients)
        db.flush()

        intakes = [
            EmergencyIntake(
                patient_id=patients[0].id,
                arrival_mode="ambulance",
                chief_complaint="crushing chest pain radiating to left arm, diaphoresis",
                temperature=98.4,
                heartrate=118,
                resprate=22,
                o2sat=94,
                sbp=92,
                dbp=58,
                pain="9",
                predicted_acuity=2,
                acuity_label="Emergent",
                confidence=0.71,
                disposition="treatment",
                status="active",
                created_at=datetime.utcnow() - timedelta(minutes=22),
            ),
            EmergencyIntake(
                patient_id=patients[1].id,
                arrival_mode="walk-in",
                chief_complaint="fever and productive cough for 3 days",
                temperature=101.2,
                heartrate=96,
                resprate=18,
                o2sat=97,
                sbp=128,
                dbp=82,
                pain="3",
                predicted_acuity=3,
                acuity_label="Urgent",
                confidence=0.64,
                disposition="waiting",
                status="active",
                created_at=datetime.utcnow() - timedelta(minutes=48),
            ),
            EmergencyIntake(
                patient_id=patients[2].id,
                arrival_mode="walk-in",
                chief_complaint="ankle sprain after fall, able to bear weight",
                temperature=98.6,
                heartrate=78,
                resprate=14,
                o2sat=99,
                sbp=118,
                dbp=76,
                pain="4",
                predicted_acuity=4,
                acuity_label="Less Urgent",
                confidence=0.58,
                disposition="waiting",
                status="active",
                created_at=datetime.utcnow() - timedelta(minutes=12),
            ),
        ]
        db.add_all(intakes)

    db.commit()
