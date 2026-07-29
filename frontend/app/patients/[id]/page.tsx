"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { AISummary, Intake, api } from "@/lib/api";

export default function PatientDetailPage() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  const [intake, setIntake] = useState<Intake | null>(null);
  const [summary, setSummary] = useState<AISummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) return;
    api
      .intake(id)
      .then(setIntake)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
  }, [id]);

  async function generate() {
    setLoading(true);
    setError("");
    try {
      const s = await api.summarize(id);
      setSummary(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Summary failed");
    } finally {
      setLoading(false);
    }
  }

  if (!intake && !error) return <p>Loading case…</p>;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>
            {intake?.patient
              ? `${intake.patient.first_name} ${intake.patient.last_name}`
              : `Intake #${id}`}
          </h1>
          <p>
            MRN {intake?.patient?.mrn ?? "—"} · Arrival {intake?.arrival_mode} · ESI{" "}
            {intake?.predicted_acuity ?? "—"} ({intake?.acuity_label ?? "n/a"})
          </p>
        </div>
        <button className="btn btn-primary" onClick={generate} disabled={loading}>
          {loading ? "Running LangChain RAG…" : "Generate AI summary"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {intake && (
        <div className="grid-2">
          <section className="panel stack">
            <h2 style={{ marginTop: 0, fontFamily: "var(--display)" }}>Clinical intake</h2>
            <p>
              <strong>Chief complaint:</strong> {intake.chief_complaint}
            </p>
            <p>
              Vitals — Temp {intake.temperature ?? "—"} · HR {intake.heartrate ?? "—"} · RR{" "}
              {intake.resprate ?? "—"} · SpO₂ {intake.o2sat ?? "—"} · BP {intake.sbp ?? "—"}/
              {intake.dbp ?? "—"} · Pain {intake.pain ?? "—"}
            </p>
            <p>
              Disposition: {intake.disposition} · Status: {intake.status}
              {intake.confidence != null
                ? ` · Model confidence ${Math.round(intake.confidence * 100)}%`
                : ""}
            </p>
            {intake.notes && <p>Notes: {intake.notes}</p>}
          </section>

          <section className="panel stack">
            <h2 style={{ marginTop: 0, fontFamily: "var(--display)" }}>
              AI-assisted summary
            </h2>
            {!summary && (
              <p style={{ color: "var(--muted)" }}>
                Uses LangChain retrieval over ED protocol knowledge, then generates a
                patient summary and workflow recommendations.
              </p>
            )}
            {summary && (
              <>
                <div className="mono">{summary.summary_text}</div>
                {summary.recommendations && (
                  <>
                    <h3 style={{ fontFamily: "var(--display)", marginBottom: 0 }}>
                      Recommendations
                    </h3>
                    <div className="mono">{summary.recommendations}</div>
                  </>
                )}
                <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                  Model: {summary.model_name}
                </p>
              </>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
