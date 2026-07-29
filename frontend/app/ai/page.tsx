"use client";

import { FormEvent, useState } from "react";
import { api } from "@/lib/api";

export default function AIPage() {
  const [result, setResult] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setError("");
    const fd = new FormData(e.currentTarget);
    const acuityRaw = String(fd.get("acuity") || "").trim();
    try {
      const res = await api.recommend({
        chief_complaint: String(fd.get("chief_complaint") || ""),
        acuity: acuityRaw ? Number(acuityRaw) : undefined,
        context: String(fd.get("context") || "") || undefined,
      });
      setResult(res.recommendations);
      setSources(res.sources);
      setModel(res.model_name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recommendation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>AI workflow advisor</h1>
          <p>
            LangChain RAG over clinical protocol documents for ED workflow
            recommendations.
          </p>
        </div>
      </div>

      <div className="grid-2">
        <form className="panel" onSubmit={onSubmit}>
          <div className="form-grid" style={{ gridTemplateColumns: "1fr" }}>
            <div className="field">
              <label>Chief complaint</label>
              <textarea
                name="chief_complaint"
                required
                rows={4}
                placeholder="shortness of breath and wheezing"
              />
            </div>
            <div className="field">
              <label>Predicted / known ESI (optional)</label>
              <input name="acuity" type="number" min={1} max={5} placeholder="2" />
            </div>
            <div className="field">
              <label>Extra context</label>
              <textarea name="context" rows={3} placeholder="Known asthma, SpO2 91%" />
            </div>
          </div>
          <div className="actions">
            <button className="btn btn-primary" disabled={loading} type="submit">
              {loading ? "Retrieving…" : "Recommend workflow"}
            </button>
          </div>
          {error && <p className="error">{error}</p>}
        </form>

        <section className="panel stack">
          <h2 style={{ marginTop: 0, fontFamily: "var(--display)" }}>Output</h2>
          {result ? (
            <>
              <div className="mono">{result}</div>
              <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                Sources: {sources.join(", ") || "knowledge base"} · {model}
              </p>
            </>
          ) : (
            <p style={{ color: "var(--muted)" }}>
              Results appear here after retrieval + generation.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
