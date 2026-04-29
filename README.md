# Triage Acuity Predictor

End-to-end web app that predicts the **Emergency Severity Index (ESI 1–5)** from
a patient's vital signs and free-text chief complaint. Trained on the
[MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) `triage.csv` table.

## Stack

- **Model:** LightGBM multi-class classifier on combined numeric vitals
  (temperature, heart rate, respiratory rate, SpO₂, BP, pain, derived shock
  index / pulse pressure / MAP) and TF-IDF features over the chief complaint.
- **Backend:** FastAPI + Uvicorn.
- **Frontend:** Server-rendered Jinja template with a vanilla JS form that
  POSTs to `/predict`.

## Project Layout

```
.
├── app/
│   ├── main.py            FastAPI app (GET /, POST /predict, GET /health)
│   ├── schemas.py         Pydantic request/response schemas
│   ├── static/            CSS + JS for the UI
│   └── templates/         index.html
├── data/raw/              triage.csv, vitalsign.csv (raw MIMIC-IV-ED)
├── models/                triage_model_v1.pkl (saved bundle)
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py   shared feature engineering for train + inference
│   └── model.py           training entry point
├── requirements.txt
└── README.md
```

## Setup

```powershell
pip install -r requirements.txt
```

## Train

```powershell
python -m src.model
```

Saves the bundle to `models/triage_model_v1.pkl`. Use `--sample 20000` for a
fast smoke test.

## Run the Web App

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## API

`POST /predict` with JSON body:

```json
{
  "temperature": 98.7,
  "heartrate": 132,
  "resprate": 24,
  "o2sat": 91,
  "sbp": 88,
  "dbp": 54,
  "pain": "9",
  "chiefcomplaint": "crushing chest pain radiating to left arm"
}
```

Response:

```json
{
  "acuity": 1,
  "label": "Resuscitation",
  "description": "Immediately life-threatening — needs lifesaving intervention now.",
  "confidence": 0.78,
  "probabilities": { "1": 0.78, "2": 0.16, "3": 0.05, "4": 0.01, "5": 0.0 },
  "top_features": ["chest pain", "shortness", "left arm"]
}
```

## Disclaimer

This is a research/educational project. The model is **not** a clinical device
and must not be used to make real triage decisions.
