"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, OpsDashboard } from "@/lib/api";

function acuityClass(n?: number | null) {
  if (!n) return "badge";
  if (n <= 2) return "badge hot";
  if (n === 3) return "badge warn";
  return "badge";
}

export default function DashboardPage() {
  const [data, setData] = useState<OpsDashboard | null>(null);
  const [health, setHealth] = useState<string>("checking…");
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [dash, h] = await Promise.all([api.dashboard(), api.health()]);
        if (!alive) return;
        setData(dash);
        setHealth(`${h.status}${h.model_kind ? ` · ${h.model_kind}` : ""}`);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Failed to load dashboard");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Operations board</h1>
          <p>
            Live emergency census, acuity mix, and throughput signals from the
            Smartflow backend.
          </p>
        </div>
        <div>
          <span className="status-dot" />
          {health}
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      {data && (
        <>
          <div className="metrics">
            <div className="metric">
              <div className="label">Active intakes</div>
              <div className="value">{data.active_intakes}</div>
            </div>
            <div className="metric">
              <div className="label">Waiting</div>
              <div className="value">{data.waiting_count}</div>
            </div>
            <div className="metric">
              <div className="label">Avg ESI</div>
              <div className="value">{data.avg_acuity ?? "—"}</div>
            </div>
            <div className="metric">
              <div className="label">Tracked metrics</div>
              <div className="value">{data.metrics.length}</div>
            </div>
          </div>

          <div className="grid-2">
            <section className="panel">
              <h2 style={{ marginTop: 0, fontFamily: "var(--display)" }}>Recent intakes</h2>
              <table>
                <thead>
                  <tr>
                    <th>Patient</th>
                    <th>Complaint</th>
                    <th>ESI</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent_intakes.map((row) => (
                    <tr key={row.id}>
                      <td>
                        {row.patient
                          ? `${row.patient.last_name}, ${row.patient.first_name}`
                          : `#${row.patient_id}`}
                      </td>
                      <td>{row.chief_complaint.slice(0, 42)}{row.chief_complaint.length > 42 ? "…" : ""}</td>
                      <td>
                        <span className={acuityClass(row.predicted_acuity)}>
                          {row.predicted_acuity ?? "—"} {row.acuity_label ?? ""}
                        </span>
                      </td>
                      <td>
                        <Link href={`/patients/${row.id}`}>Open</Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="panel">
              <h2 style={{ marginTop: 0, fontFamily: "var(--display)" }}>Ops metrics</h2>
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Dept</th>
                  </tr>
                </thead>
                <tbody>
                  {data.metrics.map((m) => (
                    <tr key={m.id}>
                      <td>{m.metric_label}</td>
                      <td>
                        {m.value}
                        <span className="unit"> {m.unit}</span>
                      </td>
                      <td>{m.department}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
