"""Quick smoke test against the running server."""
import json
import urllib.request

CASES = [
    ("STEMI-like", {"temperature": 98.7, "heartrate": 132, "resprate": 24, "o2sat": 91,
                     "sbp": 88, "dbp": 54, "pain": "9",
                     "chiefcomplaint": "crushing chest pain radiating to left arm, shortness of breath, diaphoresis"}),
    ("Sprain",     {"temperature": 98.2, "heartrate": 76, "resprate": 14, "o2sat": 99,
                     "sbp": 118, "dbp": 74, "pain": "2",
                     "chiefcomplaint": "ankle sprain after fall, mild swelling"}),
    ("Sepsis",     {"temperature": 103.1, "heartrate": 118, "resprate": 22, "o2sat": 94,
                     "sbp": 100, "dbp": 60, "pain": "6",
                     "chiefcomplaint": "high fever, productive cough, weakness, altered mental status"}),
    ("Abd pain",   {"temperature": 99.4, "heartrate": 92, "resprate": 18, "o2sat": 97,
                     "sbp": 124, "dbp": 78, "pain": "5",
                     "chiefcomplaint": "abdominal pain, nausea, vomiting for 2 days"}),
    ("Cold",       {"temperature": 99.1, "heartrate": 78, "resprate": 14, "o2sat": 99,
                     "sbp": 122, "dbp": 76, "pain": "1",
                     "chiefcomplaint": "runny nose, mild sore throat for 3 days"}),
]


def main():
    for name, payload in CASES:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/predict",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        res = json.loads(urllib.request.urlopen(req).read())
        probs = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(res["probabilities"].items()))
        print(f"{name:>12s} -> ESI {res['acuity']} ({res['label']}) "
              f"conf={res['confidence']:.2f}  [{probs}]  top={res['top_features']}")


if __name__ == "__main__":
    main()
