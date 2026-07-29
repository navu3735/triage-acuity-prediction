# Smartflow — AI Healthcare Platform

Full-stack AI healthcare platform for emergency intake, operational ED metrics,
ESI acuity prediction (TensorFlow / scikit-learn), and LangChain RAG summaries.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | React + Next.js |
| Backend | FastAPI REST APIs |
| Database | PostgreSQL (Docker) / SQLite fallback |
| ML | TensorFlow (optional) + scikit-learn ESI model |
| LLM | LangChain workflows + RAG over ED protocols |
| Deploy | Docker Compose microservices + GitHub Actions CI/CD |

## Setup

```powershell
# Vercel / triage-only
pip install -r requirements.txt

# Full Smartflow platform (intake, ops, LangChain RAG)
pip install -r requirements-platform.txt
```

### 1. Backend API

```powershell
pip install -r requirements-platform.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API docs: http://127.0.0.1:8000/docs  
Legacy triage UI: http://127.0.0.1:8000/

### 2. Next.js frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### Docker (Postgres + API + Web)

```powershell
docker compose up --build
```

- Web: http://localhost:3000  
- API: http://localhost:8000  
- Postgres: `localhost:5432` (`smartflow` / `smartflow`)

Optional: set `OPENAI_API_KEY` for live LLM generation. Without it, LangChain
still runs retrieval + a local synthesizer so demos work offline.

## REST surface

- `POST /predict` — ESI acuity from vitals + complaint (TensorFlow/sklearn)
- `POST /api/intake` — emergency intake persistence + prediction
- `GET /api/intake` — list intakes
- `GET /api/ops/dashboard` — operational healthcare dashboard data
- `GET /api/ops/metrics` — throughput / census metrics
- `POST /api/ai/summarize` — LangChain RAG patient summary
- `POST /api/ai/recommend` — workflow recommendations

## Project layout

```
.
├── frontend/                 Next.js + React UI
├── app/                      FastAPI + RAG + intake/ops APIs
├── src/                      ML training / preprocessing
├── models/                   Trained ESI models
├── docker-compose.yml
└── .github/workflows/ci.yml
```
