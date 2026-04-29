/* eslint-disable */

// ---------- normal adult ranges ----------
const VITAL_RANGES = {
  temperature: { normal: [97.0, 99.5], borderline: [96.0, 100.4] },
  heartrate:   { normal: [60, 100],    borderline: [50, 110] },
  resprate:    { normal: [12, 20],     borderline: [10, 24] },
  o2sat:       { normal: [95, 100],    borderline: [92, 100] },
  sbp:         { normal: [90, 140],    borderline: [85, 160] },
  dbp:         { normal: [60, 90],     borderline: [55, 100] },
};

function flagState(name, val) {
  if (val === null || val === undefined || isNaN(val)) return null;
  const r = VITAL_RANGES[name];
  if (!r) return null;
  if (val >= r.normal[0] && val <= r.normal[1]) return "normal";
  if (val >= r.borderline[0] && val <= r.borderline[1]) return "borderline";
  return "abnormal";
}

// ---------- ESI clinical context ----------
const ESI_INFO = {
  1: {
    label: "Resuscitation",
    target: "See **immediately** · resuscitation bay",
    description: "Immediately life-threatening. Activate code team, secure airway/breathing/circulation, full monitoring.",
  },
  2: {
    label: "Emergent",
    target: "See within **~10 minutes** · acute care bed",
    description: "High risk of deterioration. Immediate provider eval, IV access, labs/ECG, anticipate aggressive workup.",
  },
  3: {
    label: "Urgent",
    target: "See within **~30–60 min** · standard ED bed",
    description: "Stable but multiple resources expected (labs, imaging, IV meds). Reassess vitals q1h.",
  },
  4: {
    label: "Less Urgent",
    target: "See within **~1–2 hours** · fast-track / minor care",
    description: "Single resource expected. Suitable for fast-track or minor-care area.",
  },
  5: {
    label: "Non-Urgent",
    target: "Routine · longest acceptable wait",
    description: "No ED resources expected. Often appropriate for outpatient clinic / direct discharge planning.",
  },
};

const QUICK_PICKS = [
  "Chest pain",
  "Shortness of breath",
  "Abdominal pain",
  "Headache",
  "Fever",
  "Back pain",
  "Trauma / fall",
  "Altered mental status",
  "Vomiting",
  "Laceration",
];

// ---------- DOM refs ----------
const form = document.getElementById("triage-form");
const submitBtn = document.getElementById("submit-btn");
const clearBtn = document.getElementById("clear-btn");
const ccField = document.getElementById("cc");
const painScale = document.getElementById("pain-scale");
const painValue = document.getElementById("pain-value");
const quickpicks = document.getElementById("quickpicks");

const resultWrap = document.getElementById("result-wrap");
const resultCard = document.getElementById("result-card");
const esiNum = document.getElementById("esi-num");
const resultLabel = document.getElementById("result-label");
const resultTarget = document.getElementById("result-target");
const resultDescription = document.getElementById("result-description");
const resultConfidence = document.getElementById("result-confidence");
const probaBars = document.getElementById("proba-bars");
const featuresBlock = document.getElementById("features-block");
const featurePills = document.getElementById("feature-pills");

const derivedRow = document.getElementById("derived-row");
const dMap = document.getElementById("d-map");
const dPp = document.getElementById("d-pp");
const dSi = document.getElementById("d-si");

// ---------- pain scale ----------
for (let i = 0; i <= 10; i++) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "pain-btn";
  btn.dataset.score = i;
  btn.textContent = i;
  btn.setAttribute("role", "radio");
  btn.setAttribute("aria-label", `Pain ${i}`);
  btn.addEventListener("click", () => setPain(i));
  painScale.appendChild(btn);
}
function setPain(score) {
  painValue.value = score === null ? "" : String(score);
  for (const btn of painScale.children) {
    btn.classList.toggle("selected", score !== null && Number(btn.dataset.score) === Number(score));
  }
}

// ---------- quick picks ----------
for (const tag of QUICK_PICKS) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "qp-btn";
  btn.textContent = tag;
  btn.addEventListener("click", () => {
    const cur = ccField.value.trim();
    const lower = tag.toLowerCase();
    if (cur.toLowerCase().includes(lower)) {
      ccField.focus();
      return;
    }
    ccField.value = cur ? `${cur}, ${lower}` : lower;
    ccField.focus();
  });
  quickpicks.appendChild(btn);
}

// ---------- live vital flagging + derived ----------
function updateVitalFlags() {
  const data = {};
  for (const f of Object.keys(VITAL_RANGES)) {
    const inp = form.querySelector(`[name="${f}"]`);
    const val = inp.value === "" ? null : Number(inp.value);
    data[f] = val;
    const state = flagState(f, val);
    const card = inp.closest(".monitor-card");
    const flag = card.querySelector(".m-flag");
    if (state) flag.dataset.state = state;
    else delete flag.dataset.state;
    card.classList.toggle("abnormal", state === "abnormal");
    card.classList.toggle("borderline", state === "borderline");
  }
  const { sbp, dbp, heartrate } = data;
  const haveBP = Number.isFinite(sbp) && Number.isFinite(dbp);
  const haveHR = haveBP && Number.isFinite(heartrate) && sbp > 0;
  if (haveBP) {
    derivedRow.hidden = false;
    dMap.textContent = (dbp + (sbp - dbp) / 3).toFixed(0);
    dPp.textContent = (sbp - dbp).toFixed(0);
    dSi.textContent = haveHR ? (heartrate / sbp).toFixed(2) : "—";
  } else {
    derivedRow.hidden = true;
  }
}
form.addEventListener("input", updateVitalFlags);

// ---------- submit ----------
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitForm();
});

async function submitForm() {
  submitBtn.disabled = true;
  submitBtn.querySelector(".btn-label").textContent = "Predicting…";

  const formData = new FormData(form);
  const payload = {};
  for (const [key, raw] of formData.entries()) {
    const value = String(raw).trim();
    if (!value) continue;
    if (["chiefcomplaint", "pain"].includes(key)) {
      payload[key] = value;
    } else {
      const num = Number(value);
      payload[key] = Number.isFinite(num) ? num : null;
    }
  }
  if (Object.keys(payload).length === 0) {
    alert("Enter at least one field before predicting.");
    resetSubmit();
    return;
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || "Prediction failed");
    }
    const data = await response.json();
    renderResult(data);
  } catch (err) {
    alert(err.message);
  } finally {
    resetSubmit();
  }
}
function resetSubmit() {
  submitBtn.disabled = false;
  submitBtn.querySelector(".btn-label").textContent = "Predict acuity";
}

// ---------- render ----------
function renderResult(data) {
  const info = ESI_INFO[data.acuity] || { label: "Unknown", target: "", description: "" };

  // re-trigger animation by toggling the wrap hidden state
  resultWrap.hidden = false;
  resultWrap.style.animation = "none";
  // force reflow so animation restarts
  void resultWrap.offsetWidth;
  resultWrap.style.animation = "";

  resultCard.className = `result-card a${data.acuity}`;
  esiNum.textContent = data.acuity;
  resultLabel.textContent = info.label;
  resultTarget.innerHTML = formatBold(info.target);
  resultDescription.textContent = info.description;
  resultConfidence.textContent = `${(data.confidence * 100).toFixed(0)}%`;

  const entries = Object.entries(data.probabilities)
    .map(([k, v]) => [Number(k), v])
    .sort((a, b) => a[0] - b[0]);
  probaBars.innerHTML = "";
  for (const [acuity, prob] of entries) {
    const row = document.createElement("div");
    row.className = "proba-row";
    row.innerHTML = `
      <div class="pname">${acuity}</div>
      <div class="proba-bar-track"><div class="proba-bar-fill a${acuity}" style="width: ${(prob * 100).toFixed(1)}%"></div></div>
      <div class="pval">${(prob * 100).toFixed(1)}%</div>
    `;
    probaBars.appendChild(row);
  }

  if (data.top_features && data.top_features.length) {
    featuresBlock.hidden = false;
    featurePills.innerHTML = "";
    for (const feat of data.top_features) {
      const pill = document.createElement("span");
      pill.className = "feat-pill";
      pill.textContent = feat;
      featurePills.appendChild(pill);
    }
  } else {
    featuresBlock.hidden = true;
  }

  resultWrap.scrollIntoView({ behavior: "smooth", block: "start" });
}

function formatBold(s) {
  return s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

// ---------- clear / new patient ----------
function clearForm() {
  form.reset();
  setPain(null);
  updateVitalFlags();
  resultWrap.hidden = true;
  const first = form.querySelector("input");
  if (first) first.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
clearBtn.addEventListener("click", clearForm);

// ---------- keyboard ----------
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    submitForm();
  } else if (e.key === "Escape") {
    e.preventDefault();
    clearForm();
  }
});

// ---------- model status badge ----------
async function loadHealth() {
  const pill = document.getElementById("model-status");
  const text = document.getElementById("model-status-text");
  try {
    const r = await fetch("/health");
    const data = await r.json();
    if (data.status === "ok") {
      const acc = data.metrics?.accuracy;
      const top2 = data.metrics?.top_2_accuracy;
      pill.className = "status-pill ok";
      text.textContent = `model online · ${(acc * 100).toFixed(1)}% acc`;
      const accStat = document.getElementById("acc-stat");
      const top2Stat = document.getElementById("top2-stat");
      if (accStat) accStat.textContent = `${(acc * 100).toFixed(1)}%`;
      if (top2Stat) top2Stat.textContent = `${(top2 * 100).toFixed(1)}%`;
    } else {
      pill.className = "status-pill degraded";
      text.textContent = "model offline";
    }
  } catch {
    pill.className = "status-pill degraded";
    text.textContent = "unreachable";
  }
}

// ---------- init ----------
loadHealth();
updateVitalFlags();
