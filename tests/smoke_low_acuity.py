"""Probe how the model handles obviously low-acuity complaints."""
import json
import urllib.request

CASES = [
    ("cough only",            {"chiefcomplaint": "cough"}),
    ("cough+normals",         {"chiefcomplaint": "cough", "temperature": 98.6,
                                "heartrate": 75, "resprate": 14, "o2sat": 99,
                                "sbp": 120, "dbp": 78, "pain": "1"}),
    ("mild cough",            {"chiefcomplaint": "mild cough for 2 days", "temperature": 98.6,
                                "heartrate": 75, "resprate": 14, "o2sat": 99,
                                "sbp": 120, "dbp": 78, "pain": "1"}),
    ("sore throat",           {"chiefcomplaint": "sore throat", "temperature": 98.6,
                                "heartrate": 75, "resprate": 14, "o2sat": 99,
                                "sbp": 120, "dbp": 78}),
    ("rash",                  {"chiefcomplaint": "small rash on arm"}),
    ("med refill",            {"chiefcomplaint": "med refill"}),
    ("empty",                 {"chiefcomplaint": ""}),
    ("just vitals normal",    {"temperature": 98.6, "heartrate": 75, "resprate": 14,
                                "o2sat": 99, "sbp": 120, "dbp": 78, "pain": "0"}),
]


def main():
    for name, payload in CASES:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8000/predict",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            res = json.loads(urllib.request.urlopen(req).read())
            probs = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(res["probabilities"].items()))
            print(f"{name:>22s} -> ESI {res['acuity']} ({res['label']}) "
                  f"conf={res['confidence']:.2f}  [{probs}]  top={res['top_features']}")
        except Exception as exc:
            print(f"{name:>22s} -> ERROR {exc}")


if __name__ == "__main__":
    main()
