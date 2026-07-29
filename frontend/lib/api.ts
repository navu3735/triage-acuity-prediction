export type Patient = {
  id: number;
  mrn: string;
  first_name: string;
  last_name: string;
  date_of_birth?: string | null;
  sex?: string | null;
  created_at: string;
};

export type Intake = {
  id: number;
  patient_id: number;
  arrival_mode: string;
  chief_complaint: string;
  temperature?: number | null;
  heartrate?: number | null;
  resprate?: number | null;
  o2sat?: number | null;
  sbp?: number | null;
  dbp?: number | null;
  pain?: string | null;
  predicted_acuity?: number | null;
  acuity_label?: string | null;
  confidence?: number | null;
  disposition: string;
  status: string;
  notes?: string | null;
  created_at: string;
  patient?: Patient | null;
};

export type Metric = {
  id: number;
  metric_key: string;
  metric_label: string;
  value: number;
  unit: string;
  department: string;
  recorded_at: string;
};

export type OpsDashboard = {
  active_intakes: number;
  waiting_count: number;
  avg_acuity?: number | null;
  acuity_distribution: Record<string, number>;
  metrics: Metric[];
  recent_intakes: Intake[];
};

export type AISummary = {
  id: number;
  intake_id: number;
  summary_text: string;
  recommendations: string;
  sources: string;
  model_name: string;
  created_at: string;
};

const API_BASE =
  typeof window === "undefined"
    ? process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
    : process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  dashboard: () => request<OpsDashboard>("/api/ops/dashboard"),
  intakes: () => request<Intake[]>("/api/intake"),
  intake: (id: number) => request<Intake>(`/api/intake/${id}`),
  createIntake: (body: unknown) =>
    request<Intake>("/api/intake", { method: "POST", body: JSON.stringify(body) }),
  summarize: (intake_id: number) =>
    request<AISummary>("/api/ai/summarize", {
      method: "POST",
      body: JSON.stringify({ intake_id, include_recommendations: true }),
    }),
  recommend: (body: { chief_complaint: string; acuity?: number; context?: string }) =>
    request<{ recommendations: string; sources: string[]; model_name: string }>(
      "/api/ai/recommend",
      { method: "POST", body: JSON.stringify(body) }
    ),
  health: () => request<{ status: string; model_kind?: string }>("/health"),
};
