"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { api, Intake } from "@/lib/api";

export default function IntakePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [created, setCreated] = useState<Intake | null>(null);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setCreated(null);
    const fd = new FormData(e.currentTarget);
    const num = (key: string) => {
      const v = String(fd.get(key) || "").trim();
      return v ? Number(v) : null;
    };
    const body = {
      patient: {
        first_name: String(fd.get("first_name") || "").trim(),
        last_name: String(fd.get("last_name") || "").trim(),
        date_of_birth: String(fd.get("date_of_birth") || "").trim() || null,
        sex: String(fd.get("sex") || "").trim() || null,
      },
      arrival_mode: String(fd.get("arrival_mode") || "walk-in"),
      chief_complaint: String(fd.get("chief_complaint") || "").trim(),
      temperature: num("temperature"),
      heartrate: num("heartrate"),
      resprate: num("resprate"),
      o2sat: num("o2sat"),
      sbp: num("sbp"),
      dbp: num("dbp"),
      pain: String(fd.get("pain") || "").trim() || null,
      notes: String(fd.get("notes") || "").trim() || null,
      run_prediction: true,
    };

    try {
      const intake = await api.createIntake(body);
      setCreated(intake);
      e.currentTarget.reset();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Intake failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Emergency intake</h1>
          <p>
            Capture arrival + vitals, persist to PostgreSQL/SQLite, and run the
            TensorFlow/sklearn ESI acuity model.
          </p>
        </div>
      </div>

      <form className="panel" onSubmit={onSubmit}>
        <div className="form-grid">
          <div className="field">
            <label>First name</label>
            <input name="first_name" required placeholder="Maya" />
          </div>
          <div className="field">
            <label>Last name</label>
            <input name="last_name" required placeholder="Chen" />
          </div>
          <div className="field">
            <label>Sex</label>
            <select name="sex" defaultValue="">
              <option value="">—</option>
              <option value="F">F</option>
              <option value="M">M</option>
              <option value="X">X</option>
            </select>
          </div>
          <div className="field">
            <label>Date of birth</label>
            <input name="date_of_birth" placeholder="YYYY-MM-DD" />
          </div>
          <div className="field">
            <label>Arrival mode</label>
            <select name="arrival_mode" defaultValue="walk-in">
              <option value="walk-in">Walk-in</option>
              <option value="ambulance">Ambulance</option>
              <option value="transfer">Transfer</option>
            </select>
          </div>
          <div className="field">
            <label>Pain (0–10)</label>
            <input name="pain" placeholder="7" />
          </div>
          <div className="field">
            <label>Temp °F</label>
            <input name="temperature" type="number" step="0.1" placeholder="98.6" />
          </div>
          <div className="field">
            <label>Heart rate</label>
            <input name="heartrate" type="number" placeholder="80" />
          </div>
          <div className="field">
            <label>Resp rate</label>
            <input name="resprate" type="number" placeholder="16" />
          </div>
          <div className="field">
            <label>SpO₂ %</label>
            <input name="o2sat" type="number" placeholder="98" />
          </div>
          <div className="field">
            <label>SBP</label>
            <input name="sbp" type="number" placeholder="120" />
          </div>
          <div className="field">
            <label>DBP</label>
            <input name="dbp" type="number" placeholder="80" />
          </div>
          <div className="field span-3">
            <label>Chief complaint</label>
            <textarea
              name="chief_complaint"
              required
              rows={3}
              placeholder="crushing chest pain radiating to left arm"
            />
          </div>
          <div className="field span-3">
            <label>Notes</label>
            <textarea name="notes" rows={2} placeholder="Optional clinical notes" />
          </div>
        </div>

        <div className="actions">
          <button className="btn btn-primary" disabled={loading} type="submit">
            {loading ? "Running triage…" : "Submit intake + predict"}
          </button>
        </div>

        {error && <p className="error">{error}</p>}
        {created && (
          <div className="success">
            Created intake #{created.id} · ESI {created.predicted_acuity}{" "}
            ({created.acuity_label}) · confidence{" "}
            {created.confidence != null ? Math.round(created.confidence * 100) : "—"}%
            <div className="actions">
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => router.push(`/patients/${created.id}`)}
              >
                Open case + AI summary
              </button>
            </div>
          </div>
        )}
      </form>
    </div>
  );
}
