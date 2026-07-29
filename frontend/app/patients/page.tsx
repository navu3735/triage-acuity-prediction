"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Intake } from "@/lib/api";

export default function PatientsPage() {
  const [rows, setRows] = useState<Intake[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .intakes()
      .then(setRows)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
  }, []);

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Active cases</h1>
          <p>Emergency intakes stored in the operational healthcare database.</p>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <section className="panel">
        <table>
          <thead>
            <tr>
              <th>MRN</th>
              <th>Patient</th>
              <th>Complaint</th>
              <th>ESI</th>
              <th>Disposition</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>{row.patient?.mrn ?? "—"}</td>
                <td>
                  {row.patient
                    ? `${row.patient.last_name}, ${row.patient.first_name}`
                    : `#${row.patient_id}`}
                </td>
                <td>{row.chief_complaint}</td>
                <td>
                  {row.predicted_acuity ?? "—"} {row.acuity_label ?? ""}
                </td>
                <td>{row.disposition}</td>
                <td>
                  <Link href={`/patients/${row.id}`}>Details</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
