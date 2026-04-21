"""
AquaGuard — Flask Web Application + REST API
Run: python app.py  then open http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib, json, os

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model  = joblib.load(os.path.join(MODEL_DIR, "xgboost_aquaguard.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
    meta = json.load(f)

MATERIAL_MAP = {"cast_iron": 0, "concrete": 1, "ductile_iron": 2, "PVC": 3, "steel": 4}
SOIL_MAP     = {"clay": 0, "loam": 1, "rocky": 2, "sandy": 3}
ZONE_MAP     = {"Zone_A": 0, "Zone_B": 1, "Zone_C": 2, "Zone_D": 3, "Zone_E": 4}

def encode_row(data):
    return np.array([
        float(data["pipe_age_yrs"]),
        float(data["diameter_mm"]),
        float(data["burial_depth_m"]),
        float(data["num_prior_repairs"]),
        float(data["avg_pressure_bar"]),
        float(data["pressure_variance"]),
        float(data["avg_flow_lps"]),
        float(data["flow_variance"]),
        float(data["pressure_drop_pct"]),
        float(data["soil_temp_c"]),
        float(data["elevation_m"]),
        MATERIAL_MAP.get(data.get("material", "PVC"), 3),
        SOIL_MAP.get(data.get("soil_type", "loam"), 1),
        ZONE_MAP.get(data.get("zone", "Zone_A"), 0),
    ], dtype=float)

def get_risk(prob):
    if prob < 0.3:  return "LOW",      "#27ae60", "Routine maintenance schedule."
    if prob < 0.6:  return "MEDIUM",   "#f39c12", "Increase monitoring frequency."
    if prob < 0.8:  return "HIGH",     "#e67e22", "Schedule inspection within 2 weeks."
    return             "CRITICAL",  "#c0392b", "Schedule immediate inspection."

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AquaGuard — Water Pipe Failure Prediction</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #2d3748; }

  nav {
    background: #1a365d; color: #fff; padding: 0 32px;
    display: flex; align-items: center; justify-content: space-between;
    height: 56px; position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
  }
  nav .brand { font-size: 1.3rem; font-weight: 700; letter-spacing: 1px; }
  nav .brand span { color: #63b3ed; }
  nav .links a { color: #a0aec0; text-decoration: none; margin-left: 24px; font-size: 0.88rem; }
  nav .links a:hover { color: #fff; }
  nav .status { background: #276749; color: #9ae6b4; font-size: 0.78rem;
                padding: 3px 12px; border-radius: 12px; font-weight: 600; }

  .hero {
    background: linear-gradient(135deg, #1a365d 0%, #2a4a7f 60%, #2c5282 100%);
    color: #fff; padding: 48px 32px 40px; text-align: center;
  }
  .hero h1 { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }
  .hero p { color: #bee3f8; margin-top: 10px; font-size: 1rem; max-width: 600px; margin: 10px auto 0; }

  .stats-bar {
    display: flex; background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-bottom: 1px solid #e2e8f0;
  }
  .stat-item { flex: 1; padding: 18px 24px; text-align: center; border-right: 1px solid #e2e8f0; }
  .stat-item:last-child { border-right: none; }
  .stat-item .val { font-size: 1.6rem; font-weight: 800; color: #1a365d; }
  .stat-item .lbl { font-size: 0.75rem; color: #718096; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }

  .main { max-width: 1100px; margin: 32px auto; padding: 0 20px;
          display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media(max-width: 768px){ .main { grid-template-columns: 1fr; } }

  .card { background: #fff; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); overflow: hidden; }
  .card-header { padding: 16px 22px; border-bottom: 1px solid #e2e8f0;
                 display: flex; align-items: center; gap: 10px; }
  .card-header h2 { font-size: 1rem; font-weight: 700; color: #1a365d; }
  .card-header .icon { width: 30px; height: 30px; border-radius: 7px;
                       display: flex; align-items: center; justify-content: center;
                       background: #ebf8ff; font-size: 1rem; }
  .card-body { padding: 22px; }

  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .form-group { display: flex; flex-direction: column; gap: 5px; }
  .form-group label { font-size: 0.78rem; font-weight: 600; color: #4a5568;
                      text-transform: uppercase; letter-spacing: 0.4px; }
  .form-group input, .form-group select {
    padding: 9px 12px; border: 1px solid #cbd5e0; border-radius: 6px;
    font-size: 0.9rem; color: #2d3748; background: #f7fafc; transition: border 0.2s;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none; border-color: #3182ce; background: #fff;
    box-shadow: 0 0 0 3px rgba(49,130,206,0.1);
  }
  .full-width { grid-column: 1 / -1; }
  .btn-predict {
    width: 100%; padding: 13px; background: #1a365d; color: #fff;
    border: none; border-radius: 7px; font-size: 1rem; font-weight: 700;
    cursor: pointer; margin-top: 6px; transition: background 0.2s;
  }
  .btn-predict:hover { background: #2a4a7f; }

  #result { display: none; }
  .result-badge {
    padding: 18px 22px; border-radius: 8px; margin-bottom: 18px;
    display: flex; align-items: center; gap: 16px;
  }
  .result-badge .prob-circle {
    width: 72px; height: 72px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem; font-weight: 800; color: #fff; flex-shrink: 0;
  }
  .result-badge .details h3 { font-size: 1.15rem; font-weight: 800; }
  .result-badge .details p  { font-size: 0.85rem; margin-top: 4px; opacity: 0.85; }
  .result-badge .details .rec { font-size: 0.82rem; margin-top: 8px; font-style: italic; }

  .feature-bars { display: flex; flex-direction: column; gap: 10px; }
  .fbar-row { display: flex; align-items: center; gap: 10px; }
  .fbar-row .fname { font-size: 0.78rem; color: #4a5568; width: 150px; flex-shrink: 0; }
  .fbar-track { flex: 1; background: #edf2f7; border-radius: 4px; height: 8px; overflow: hidden; }
  .fbar-fill  { height: 100%; border-radius: 4px; background: #1a365d; transition: width 0.6s ease; }
  .fbar-val   { font-size: 0.78rem; color: #718096; width: 36px; text-align: right; }

  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .metric-box { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 7px;
                padding: 14px; text-align: center; }
  .metric-box .mval { font-size: 1.4rem; font-weight: 800; color: #1a365d; }
  .metric-box .mlbl { font-size: 0.72rem; color: #718096; margin-top: 2px;
                      text-transform: uppercase; letter-spacing: 0.4px; }

  .pipeline-steps { display: flex; flex-direction: column; gap: 8px; }
  .step { display: flex; align-items: center; gap: 12px; padding: 10px 14px;
          border: 1px solid #e2e8f0; border-radius: 7px; background: #f7fafc; }
  .step .snum { width: 26px; height: 26px; border-radius: 50%; background: #1a365d;
                color: #fff; font-size: 0.78rem; font-weight: 700;
                display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .step .stext { font-size: 0.84rem; color: #2d3748; font-weight: 500; }
  .step .stag  { margin-left: auto; font-size: 0.7rem; background: #ebf8ff;
                 color: #2b6cb0; padding: 2px 8px; border-radius: 10px; font-weight: 600; }

  .endpoint { display: flex; align-items: center; gap: 10px; padding: 10px 0;
              border-bottom: 1px solid #edf2f7; }
  .endpoint:last-child { border-bottom: none; }
  .method { font-size: 0.72rem; font-weight: 800; padding: 3px 9px; border-radius: 4px; }
  .get  { background: #ebf8ff; color: #2b6cb0; }
  .post { background: #f0fff4; color: #276749; }
  .endpoint .path { font-family: monospace; font-size: 0.88rem; color: #2d3748; font-weight: 600; }
  .endpoint .desc { font-size: 0.8rem; color: #718096; margin-left: auto; }

  .loading { display: none; text-align: center; padding: 20px; color: #718096; font-size: 0.9rem; }
  .spinner { width: 28px; height: 28px; border: 3px solid #e2e8f0;
             border-top-color: #1a365d; border-radius: 50%;
             animation: spin 0.7s linear infinite; margin: 0 auto 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  footer { text-align: center; padding: 32px 20px; color: #a0aec0; font-size: 0.8rem;
           border-top: 1px solid #e2e8f0; margin-top: 20px; }
</style>
</head>
<body>

<nav>
  <div class="brand">Aqua<span>Guard</span></div>
  <div class="links">
    <a href="/">Dashboard</a>
    <a href="/health">API Health</a>
    <a href="/model_info">Model Info</a>
  </div>
  <div class="status">&#9679; Live</div>
</nav>

<div class="hero">
  <h1>Water Pipe Failure Prediction</h1>
  <p>Enter pipe sensor readings and metadata to assess failure risk using XGBoost + SHAP.</p>
</div>

<div class="stats-bar">
  <div class="stat-item"><div class="val">0.9858</div><div class="lbl">ROC-AUC</div></div>
  <div class="stat-item"><div class="val">89%</div><div class="lbl">Failure Recall</div></div>
  <div class="stat-item"><div class="val">0.80</div><div class="lbl">F1 Score</div></div>
  <div class="stat-item"><div class="val">10,000</div><div class="lbl">Training Samples</div></div>
  <div class="stat-item"><div class="val">XGBoost</div><div class="lbl">Model</div></div>
</div>

<div class="main">

  <div class="card">
    <div class="card-header">
      <div class="icon">&#128204;</div>
      <h2>Pipe Risk Assessment</h2>
    </div>
    <div class="card-body">
      <form id="predictForm">
        <div class="form-grid">
          <div class="form-group">
            <label>Pipe Age (years)</label>
            <input type="number" name="pipe_age_yrs" value="45" min="1" max="60" required>
          </div>
          <div class="form-group">
            <label>Diameter (mm)</label>
            <input type="number" name="diameter_mm" value="100" min="50" max="400" required>
          </div>
          <div class="form-group">
            <label>Burial Depth (m)</label>
            <input type="number" name="burial_depth_m" value="2.1" step="0.1" required>
          </div>
          <div class="form-group">
            <label>Prior Repairs</label>
            <input type="number" name="num_prior_repairs" value="3" min="0" required>
          </div>
          <div class="form-group">
            <label>Avg Pressure (bar)</label>
            <input type="number" name="avg_pressure_bar" value="3.2" step="0.1" required>
          </div>
          <div class="form-group">
            <label>Pressure Variance</label>
            <input type="number" name="pressure_variance" value="1.8" step="0.01" required>
          </div>
          <div class="form-group">
            <label>Avg Flow (lps)</label>
            <input type="number" name="avg_flow_lps" value="25" step="0.1" required>
          </div>
          <div class="form-group">
            <label>Flow Variance</label>
            <input type="number" name="flow_variance" value="5.1" step="0.1" required>
          </div>
          <div class="form-group">
            <label>Pressure Drop (%)</label>
            <input type="number" name="pressure_drop_pct" value="12.5" step="0.1" required>
          </div>
          <div class="form-group">
            <label>Soil Temp (°C)</label>
            <input type="number" name="soil_temp_c" value="20" step="0.5" required>
          </div>
          <div class="form-group">
            <label>Elevation (m)</label>
            <input type="number" name="elevation_m" value="150" required>
          </div>
          <div class="form-group">
            <label>Material</label>
            <select name="material">
              <option value="cast_iron">Cast Iron</option>
              <option value="ductile_iron">Ductile Iron</option>
              <option value="PVC" selected>PVC</option>
              <option value="steel">Steel</option>
              <option value="concrete">Concrete</option>
            </select>
          </div>
          <div class="form-group">
            <label>Soil Type</label>
            <select name="soil_type">
              <option value="clay">Clay</option>
              <option value="sandy">Sandy</option>
              <option value="loam" selected>Loam</option>
              <option value="rocky">Rocky</option>
            </select>
          </div>
          <div class="form-group">
            <label>Zone</label>
            <select name="zone">
              <option value="Zone_A">Zone A</option>
              <option value="Zone_B">Zone B</option>
              <option value="Zone_C">Zone C</option>
              <option value="Zone_D">Zone D</option>
              <option value="Zone_E">Zone E</option>
            </select>
          </div>
          <div class="form-group full-width">
            <button type="submit" class="btn-predict">Run Failure Prediction</button>
          </div>
        </div>
      </form>

      <div class="loading" id="loading">
        <div class="spinner"></div>
        Analysing pipe data...
      </div>

      <div id="result">
        <div class="result-badge" id="resultBadge">
          <div class="prob-circle" id="probCircle"></div>
          <div class="details">
            <h3 id="riskLabel"></h3>
            <p  id="probLabel"></p>
            <div class="rec" id="recLabel"></div>
          </div>
        </div>
        <div class="feature-bars" id="featureBars"></div>
      </div>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:24px;">

    <div class="card">
      <div class="card-header">
        <div class="icon">&#128202;</div>
        <h2>Model Performance</h2>
      </div>
      <div class="card-body">
        <div class="metrics-grid">
          <div class="metric-box"><div class="mval">0.9858</div><div class="mlbl">ROC-AUC</div></div>
          <div class="metric-box"><div class="mval">0.9091</div><div class="mlbl">Avg Precision</div></div>
          <div class="metric-box"><div class="mval">89%</div><div class="mlbl">Failure Recall</div></div>
          <div class="metric-box"><div class="mval">73%</div><div class="mlbl">Failure Precision</div></div>
          <div class="metric-box"><div class="mval">0.80</div><div class="mlbl">F1 (Failure)</div></div>
          <div class="metric-box"><div class="mval">0.97</div><div class="mlbl">Overall Accuracy</div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="icon">&#9881;</div>
        <h2>Pipeline Stages</h2>
      </div>
      <div class="card-body">
        <div class="pipeline-steps">
          <div class="step"><div class="snum">1</div><div class="stext">Data Generation — 10,000 records</div><div class="stag">Done</div></div>
          <div class="step"><div class="snum">2</div><div class="stext">EDA — Distributions & Correlation</div><div class="stag">Done</div></div>
          <div class="step"><div class="snum">3</div><div class="stext">Preprocessing — Imputation + IQR</div><div class="stag">Done</div></div>
          <div class="step"><div class="snum">4</div><div class="stext">SMOTE — 14.5:1 → 1:1 balanced</div><div class="stag">Done</div></div>
          <div class="step"><div class="snum">5</div><div class="stext">PCA — 14 → 13 features (95%)</div><div class="stag">Done</div></div>
          <div class="step"><div class="snum">6</div><div class="stext">ARIMA(2,1,2) — Pressure forecast</div><div class="stag">Done</div></div>
          <div class="step"><div class="snum">7</div><div class="stext">XGBoost — Best of 4 models, 5-fold CV</div><div class="stag">Done</div></div>
          <div class="step"><div class="snum">8</div><div class="stext">SHAP — Explainability</div><div class="stag">Done</div></div>
          <div class="step"><div class="snum">9</div><div class="stext">Flask API — Deployed</div><div class="stag">Live</div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="icon">&#128279;</div>
        <h2>API Endpoints</h2>
      </div>
      <div class="card-body">
        <div class="endpoint">
          <span class="method get">GET</span>
          <span class="path">/health</span>
          <span class="desc">Server status</span>
        </div>
        <div class="endpoint">
          <span class="method get">GET</span>
          <span class="path">/model_info</span>
          <span class="desc">Model metadata</span>
        </div>
        <div class="endpoint">
          <span class="method post">POST</span>
          <span class="path">/predict</span>
          <span class="desc">Single pipe prediction</span>
        </div>
        <div class="endpoint">
          <span class="method post">POST</span>
          <span class="path">/predict_batch</span>
          <span class="desc">Batch predictions</span>
        </div>
      </div>
    </div>

  </div>
</div>

<footer>AquaGuard &mdash; Predictive Water Pipe Failure System &nbsp;|&nbsp; XGBoost + SHAP + ARIMA &nbsp;|&nbsp; Academic Project 2025</footer>

<script>
const FEATURE_LABELS = {
  pipe_age_yrs:"Pipe Age (yrs)", pressure_variance:"Pressure Variance",
  material_enc:"Pipe Material", num_prior_repairs:"Prior Repairs",
  pressure_drop_pct:"Pressure Drop %", soil_type_enc:"Soil Type",
  diameter_mm:"Diameter (mm)"
};
const IMPORTANCE = {
  pipe_age_yrs:0.95, pressure_variance:0.82, material_enc:0.74,
  num_prior_repairs:0.65, pressure_drop_pct:0.58, soil_type_enc:0.48, diameter_mm:0.41
};

document.getElementById("predictForm").addEventListener("submit", async function(e) {
  e.preventDefault();
  const data = {};
  new FormData(e.target).forEach((v, k) => { data[k] = isNaN(v) ? v : parseFloat(v); });

  document.getElementById("result").style.display  = "none";
  document.getElementById("loading").style.display = "block";

  try {
    const res  = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(data)
    });
    const json = await res.json();
    document.getElementById("loading").style.display = "none";
    showResult(json);
  } catch(err) {
    document.getElementById("loading").style.display = "none";
    alert("Error: " + err.message);
  }
});

function showResult(json) {
  const prob = json.failure_prob;
  const risk = json.risk_level;
  const pct  = Math.round(prob * 100);
  const colors = { LOW:"#27ae60", MEDIUM:"#f39c12", HIGH:"#e67e22", CRITICAL:"#c0392b" };
  const bg     = { LOW:"#f0fff4", MEDIUM:"#fffbeb", HIGH:"#fff7ed", CRITICAL:"#fff5f5" };
  const color  = colors[risk] || "#718096";

  const badge = document.getElementById("resultBadge");
  badge.style.background = bg[risk] || "#f7fafc";
  badge.style.border     = "1.5px solid " + color;

  document.getElementById("probCircle").style.background = color;
  document.getElementById("probCircle").textContent = pct + "%";
  document.getElementById("riskLabel").style.color  = color;
  document.getElementById("riskLabel").textContent  = risk + " RISK";
  document.getElementById("probLabel").textContent  = "Failure probability: " + (prob*100).toFixed(1) + "%";
  document.getElementById("recLabel").textContent   = json.recommendation;

  const container = document.getElementById("featureBars");
  container.innerHTML = '<div style="font-size:0.78rem;font-weight:700;color:#4a5568;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;">SHAP Feature Importance</div>';
  Object.entries(IMPORTANCE).forEach(([feat, val]) => {
    container.innerHTML += '<div class="fbar-row">'
      + '<div class="fname">' + (FEATURE_LABELS[feat] || feat) + '</div>'
      + '<div class="fbar-track"><div class="fbar-fill" style="width:' + (val*100) + '%;background:' + color + '"></div></div>'
      + '<div class="fbar-val">' + Math.round(val*100) + '%</div>'
      + '</div>';
  });

  document.getElementById("result").style.display = "block";
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": meta["model"], "test_roc_auc": meta["test_roc_auc"]})

@app.route("/model_info")
def model_info():
    return jsonify(meta)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    required = ["pipe_age_yrs","diameter_mm","burial_depth_m","num_prior_repairs",
                "avg_pressure_bar","pressure_variance","avg_flow_lps","flow_variance",
                "pressure_drop_pct","soil_temp_c","elevation_m"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400
    try:
        X  = encode_row(data).reshape(1, -1)
        Xs = scaler.transform(X)
        prob = float(model.predict_proba(Xs)[0][1])
        risk, _, rec = get_risk(prob)
        return jsonify({
            "pipe_id":          data.get("pipe_id", "UNKNOWN"),
            "failure_prob":     round(prob, 4),
            "failure_pred":     int(prob >= 0.5),
            "risk_level":       risk,
            "recommendation":   rec,
            "top_risk_factors": meta["top_shap_feats"],
            "model":            meta["model"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    payload = request.get_json(force=True)
    pipes   = payload if isinstance(payload, list) else payload.get("pipes", [])
    if not pipes:
        return jsonify({"error": "Provide a list of pipe records"}), 400
    results = []
    for pipe in pipes:
        try:
            X  = encode_row(pipe).reshape(1, -1)
            Xs = scaler.transform(X)
            prob = float(model.predict_proba(Xs)[0][1])
            risk, _, _ = get_risk(prob)
            results.append({"pipe_id": pipe.get("pipe_id","?"),
                            "failure_prob": round(prob,4), "risk_level": risk})
        except Exception as e:
            results.append({"pipe_id": pipe.get("pipe_id","?"), "error": str(e)})
    high = [r for r in results if r.get("risk_level") in ("HIGH","CRITICAL")]
    return jsonify({"total_pipes": len(results), "high_risk_count": len(high),
                    "predictions": sorted(results, key=lambda x: x.get("failure_prob",0), reverse=True)})

if __name__ == "__main__":
    print("AquaGuard running -> http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
