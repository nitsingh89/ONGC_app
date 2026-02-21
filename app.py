import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os
import time
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh #delete

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(__file__)

DATA_FILE = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rf_models.pkl")
ANOMALY_PATH = os.path.join(BASE_DIR, "anomaly_model.pkl")
BASELINE_PATH = os.path.join(BASE_DIR, "baseline_stats.pkl")
LOG_PATH = os.path.join(BASE_DIR, "live_log.csv")


LIVE_URL = "https://markets.businessinsider.com/currencies/eth-usd"
LIVE_CLASSES = [
    "price-section__current-value",
    "price-section__absolute-value"
]

DRIFT_Z_THRESHOLD = 2.5

# =====================================================
# PAGE CONFIG + DARK THEME
# =====================================================
st.set_page_config(page_title="ONGC Predictive Maintenance", layout="wide")

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}

/* KPI Card Base */
.kpi-card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 15px rgba(0,255,255,0.2);
    transition: all 0.4s ease-in-out;
}

/* Glow animation */
@keyframes glowPulse {
    0% { box-shadow: 0 0 10px rgba(0,255,255,0.3); }
    50% { box-shadow: 0 0 25px rgba(0,255,255,0.8); }
    100% { box-shadow: 0 0 10px rgba(0,255,255,0.3); }
}

.glow {
    animation: glowPulse 2s infinite;
}

/* Severity Colors */
.normal { border: 2px solid #00ff99; }
.low { border: 2px solid #ffd700; }
.medium { border: 2px solid #ff8800; }
.high { border: 2px solid #ff0033; }

.kpi-title {
    font-size: 14px;
    opacity: 0.7;
}

.kpi-value {
    font-size: 26px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


st.title(" ONGC – Flow Meter Predictive Maintenance Dashboard")

# =====================================================
# SESSION STATE INIT
# =====================================================
st.session_state.setdefault("last_flow", None)
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("residual_history", [])

for key, default in {
    "last_flow": None,
    "last_result": None,
    "residual_history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =====================================================
# UTILITIES
# =====================================================
def fetch_live_flow():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(LIVE_URL, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    for cls in LIVE_CLASSES:
        tag = soup.find("span", class_=cls)
        if tag:
            return float(tag.text.replace(",", "").strip())
    raise ValueError("Live flow value not found")

def load_data():
    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = ["Pressure","Temperature","Flow"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df

# =====================================================
# TRAINING
# =====================================================
def train_models(df):

    models = {
        "pressure": RandomForestRegressor(n_estimators=300, random_state=42),
        "temperature": RandomForestRegressor(n_estimators=300, random_state=42),
        "flow": RandomForestRegressor(n_estimators=300, random_state=42)
    }

    models["pressure"].fit(df[["Temperature","Flow"]], df["Pressure"])
    models["temperature"].fit(df[["Pressure","Flow"]], df["Temperature"])
    models["flow"].fit(df[["Pressure","Temperature"]], df["Flow"])

    df["P_res"] = df["Pressure"] - models["pressure"].predict(df[["Temperature","Flow"]])
    df["T_res"] = df["Temperature"] - models["temperature"].predict(df[["Pressure","Flow"]])
    df["F_res"] = df["Flow"] - models["flow"].predict(df[["Pressure","Temperature"]])

    iso = IsolationForest(contamination=0.03, random_state=42)
    iso.fit(df[["P_res","T_res","F_res"]])

    baseline = {
        "mean": df[["P_res","T_res","F_res"]].mean().to_dict(),
        "std": df[["P_res","T_res","F_res"]].std().replace(0, 1e-6).to_dict()
    }

    joblib.dump(models, MODEL_PATH)
    joblib.dump(iso, ANOMALY_PATH)
    joblib.dump(baseline, BASELINE_PATH)

    return models, iso, baseline

def detect_drift(residuals, baseline):
    for col in residuals.columns:
        z = abs(
            (residuals[col].iloc[0] - baseline["mean"][col])
            / baseline["std"][col]
        )
        if z > DRIFT_Z_THRESHOLD:
            return True
    return False

# =====================================================
# LOAD / TRAIN
# =====================================================
# =====================================================
# LOAD MODELS ONLY ONCE
# =====================================================
if "models" not in st.session_state:

    df = load_data()

    if not all(map(os.path.exists, [MODEL_PATH, ANOMALY_PATH, BASELINE_PATH])):
        models, anomaly_model, baseline = train_models(df)
    else:
        models = joblib.load(MODEL_PATH)
        anomaly_model = joblib.load(ANOMALY_PATH)
        baseline = joblib.load(BASELINE_PATH)

    st.session_state.models = models
    st.session_state.anomaly_model = anomaly_model
    st.session_state.baseline = baseline

models = st.session_state.models
anomaly_model = st.session_state.anomaly_model
baseline = st.session_state.baseline




# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.image("ongc_logo.png", width=180)

st.sidebar.header("🔧 Flow Meter Inputs")
pressure = st.sidebar.number_input("Pressure", value=18.0, key="pressure_input")
temperature = st.sidebar.number_input("Temperature", value=60.0, key="temp_input")

run = st.sidebar.button("🔄 Fetch Live Flow")

st.sidebar.markdown("### ⏱ Live Control")

auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 5, 60, 10)

manual_train = st.sidebar.button("🧠 Train Models")
manual_retrain = st.sidebar.button("♻️ Retrain Models")

if manual_train:
    with st.spinner("Training models..."):
        models, anomaly_model, baseline = train_models(load_data())

    st.session_state.models = models
    st.session_state.anomaly_model = anomaly_model
    st.session_state.baseline = baseline

    st.success("✅ Training completed")


if manual_train:
    with st.spinner("Retraining models..."):
        models, anomaly_model, baseline = train_models(load_data())

    st.session_state.models = models
    st.session_state.anomaly_model = anomaly_model
    st.session_state.baseline = baseline

    st.success("✅ Retraining completed")




# =====================================================
# MAIN EXECUTION
# =====================================================

execute = run or auto_refresh

if execute:


    flow = fetch_live_flow()
    st.sidebar.success(f"Live Flow: {flow}")

    if st.session_state.last_flow != flow:
        st.session_state.last_flow = flow

        # Predictions
        p_pred = models["pressure"].predict([[temperature, flow]])[0]
        t_pred = models["temperature"].predict([[pressure, flow]])[0]
        f_pred = models["flow"].predict([[pressure, temperature]])[0]

        # Residuals
        p_res = pressure - p_pred
        t_res = temperature - t_pred
        f_res = flow - f_pred

        residuals = pd.DataFrame([[p_res,t_res,f_res]],
                                 columns=["P_res","T_res","F_res"])

        # Anomaly
        score = anomaly_model.decision_function(residuals)[0]
        flag = anomaly_model.predict(residuals)[0]

        severity = "🟢 NORMAL"
        if flag == -1:
            severity = "🟡 LOW"
            if score < -0.12: severity = "🟠 MEDIUM"
            if score < -0.25: severity = "🔴 HIGH"

        # Health Index
        baseline_std = np.mean(list(baseline["std"].values()))
        avg_residual = np.mean(np.abs([p_res,t_res,f_res]))
        health_score = max(0, min(100, 100 - (avg_residual/baseline_std)*10))

        # RUL Estimation
        st.session_state.residual_history.append(avg_residual)
        trend = 0
        rul = "Stable"

        if len(st.session_state.residual_history) > 5:
            trend = np.polyfit(
                range(len(st.session_state.residual_history)),
                st.session_state.residual_history,
                1
            )[0]
            if trend > 0:
                rul = round(health_score / (trend*10 + 1e-6),1)

        # Drift Detection
        if detect_drift(residuals, baseline):
            st.warning("⚠️ Data Drift Detected – Retraining Models")
            models, anomaly_model, baseline = train_models(load_data())

        # Logging
        log = pd.DataFrame([[datetime.now(),pressure,temperature,flow,
                             p_pred,t_pred,f_pred,
                             score,severity,health_score,rul]],
            columns=["Time","Pressure","Temperature","Flow",
                     "P_Pred","T_Pred","F_Pred",
                     "Score","Severity","HealthScore","RUL"])

        if os.path.exists(LOG_PATH):
            pd.concat([pd.read_csv(LOG_PATH),log]).to_csv(LOG_PATH,index=False)
        else:
            log.to_csv(LOG_PATH,index=False)

        st.session_state.last_result = log.iloc[0]


                #********************************#

# =====================================================
# ADVANCED PM ENGINE
# =====================================================

execute = run or auto_refresh

if execute:


    flow = fetch_live_flow()

    # ======================
    # MODEL PREDICTIONS
    # ======================
    p_pred = models["pressure"].predict([[temperature, flow]])[0]
    t_pred = models["temperature"].predict([[pressure, flow]])[0]
    f_pred = models["flow"].predict([[pressure, temperature]])[0]

    p_res = pressure - p_pred
    t_res = temperature - t_pred
    f_res = flow - f_pred

    residuals = pd.DataFrame([[p_res,t_res,f_res]],
                             columns=["P_res","T_res","F_res"])

    # ======================
    # ANOMALY + PROBABILITY
    # ======================
    score = anomaly_model.decision_function(residuals)[0]
    flag = anomaly_model.predict(residuals)[0]

    failure_prob = round(min(100, abs(score)*180),2)

    severity = "🟢 NORMAL"
    if flag == -1:
        severity = "🟡 LOW"
        if score < -0.12: severity = "🟠 MEDIUM"
        if score < -0.25: severity = "🔴 HIGH"

    # ======================
    # WEIGHTED HEALTH INDEX
    # ======================
    wP, wT, wF = 0.2, 0.3, 0.5
    weighted_res = (wP*abs(p_res) + wT*abs(t_res) + wF*abs(f_res))

    baseline_std = np.mean(list(baseline["std"].values()))
    health_score = max(0, min(100, 100 - (weighted_res/baseline_std)*12))

    # ======================
    # THRESHOLD-BASED RUL
    # ======================
    st.session_state.residual_history.append(weighted_res)

    rul = "Stable"
    if len(st.session_state.residual_history) > 8:
        trend = np.polyfit(
            range(len(st.session_state.residual_history)),
            st.session_state.residual_history,
            1
        )[0]

        if trend > 0:
            predicted_time_to_threshold = (health_score - 60) / (trend*12 + 1e-6)
            rul = round(max(0, predicted_time_to_threshold),1)

    # ======================
    # SMART DRIFT CONTROL
    # ======================
    drift_detected = detect_drift(residuals, baseline)

    if "drift_counter" not in st.session_state:
        st.session_state.drift_counter = 0

    if drift_detected:
        st.session_state.drift_counter += 1
    else:
        st.session_state.drift_counter = 0

    model_stability = max(0, 100 - st.session_state.drift_counter*10)

    if st.session_state.drift_counter >= 3:
        st.warning("⚠️ Persistent Drift – Auto Retraining Triggered")
        models, anomaly_model, baseline = train_models(load_data())
        st.session_state.drift_counter = 0

    # ======================
    # MAINTENANCE ACTION ENGINE
    # ======================
    if severity == "🔴 HIGH":
        action = "Immediate Shutdown Recommended"
    elif severity == "🟠 MEDIUM":
        action = "Inspect within 24 hrs"
    elif severity == "🟡 LOW":
        action = "Monitor Closely"
    else:
        action = "Normal Operation"

    # ======================
    # ALERT SYSTEM
    # ======================
    if severity in ["🔴 HIGH","🟠 MEDIUM"]:
        st.error(f"🚨 ALERT: {action}")

    # ======================
    # LOGGING
    # ======================
    log = pd.DataFrame([[datetime.now(),pressure,temperature,flow,
                         p_pred,t_pred,f_pred,
                         score,severity,failure_prob,
                         health_score,rul,action,model_stability]],
        columns=["Time","Pressure","Temperature","Flow",
                 "P_Pred","T_Pred","F_Pred",
                 "Score","Severity","FailureProb",
                 "HealthScore","RUL","Action","ModelStability"])

    log.to_csv(LOG_PATH, mode='a', header=not os.path.exists(LOG_PATH), index=False)

    st.session_state.last_result = log.iloc[0]


# =====================================================
# DASHBOARD DISPLAY
# =====================================================
if st.session_state.last_result is not None:

    r = st.session_state.last_result

    severity_class = "normal"
    if "LOW" in r["Severity"]:
        severity_class = "low"
    elif "MEDIUM" in r["Severity"]:
        severity_class = "medium"
    elif "HIGH" in r["Severity"]:
        severity_class = "high"

    # -------- ROW 1 --------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card glow">
            <div class="kpi-title">Pressure</div>
            <div class="kpi-value">{round(r["Pressure"],2)}</div>
            <div>Δ {round(r["Pressure"]-r["P_Pred"],2)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card glow">
            <div class="kpi-title">Temperature</div>
            <div class="kpi-value">{round(r["Temperature"],2)}</div>
            <div>Δ {round(r["Temperature"]-r["T_Pred"],2)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card glow">
            <div class="kpi-title">Flow</div>
            <div class="kpi-value">{round(r["Flow"],4)}</div>
            <div>Δ {round(r["Flow"]-r["F_Pred"],4)}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------- ROW 2 --------
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown(f"""
        <div class="kpi-card glow {severity_class}">
            <div class="kpi-title">Anomaly Level</div>
            <div class="kpi-value">{r["Severity"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class="kpi-card glow">
            <div class="kpi-title">Health Score</div>
            <div class="kpi-value">{round(r["HealthScore"],2)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class="kpi-card glow">
            <div class="kpi-title">Estimated RUL</div>
            <div class="kpi-value">{r["RUL"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
    <div class="kpi-card glow">
        <div class="kpi-title">Failure Probability</div>
        <div class="kpi-value">{r["FailureProb"]}%</div>
    </div>
    """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
    <div class="kpi-card glow">
        <div class="kpi-title">Maintenance Action</div>
        <div class="kpi-value">{r["Action"]}</div>
    </div>
    """, unsafe_allow_html=True)


    # =============================
    # Trend Visualization
    # =============================

    try:
        df_log = pd.read_csv(LOG_PATH, on_bad_lines="skip")
    except Exception:
        df_log = pd.DataFrame()
    
    required_cols = ["Flow", "F_Pred"]
    
    if df_log.empty or not all(col in df_log.columns for col in required_cols):
        st.info("No valid trend data available yet.")
    else:
        df_log["Error"] = df_log["Flow"] - df_log["F_Pred"]

    # your plotting code here



    df_log["Error"] = df_log["Flow"] - df_log["F_Pred"]

    fig = go.Figure()

    # Actual Flow
    fig.add_trace(go.Scatter(
        y=df_log["Flow"],
        mode='lines',
        name="Actual Flow",
        line=dict(width=3)
    ))

    # Predicted Flow
    fig.add_trace(go.Scatter(
        y=df_log["F_Pred"],
        mode='lines',
        name="Predicted Flow",
        line=dict(dash='dash')
    ))

    # Error (Secondary Axis)
    fig.add_trace(go.Scatter(
        y=df_log["Error"],
        mode='lines',
        name="Prediction Error",
        yaxis="y2",
        line=dict(color='red')
    ))

    fig.update_layout(
        height=500,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="white",

        yaxis=dict(title="Flow"),
        yaxis2=dict(
            title="Error",
            overlaying="y",
            side="right"
        ),

        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# AUTO REFRESH LOOP
# =====================================================
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()





