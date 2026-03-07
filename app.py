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
# SESSION STATE INIT
# =====================================================

defaults = {
    "last_flow": None,
    "last_result": None,
    "residual_history": [],
    "training_running": False,
    "training_done": False,
    "train_progress": 0,
    "engine_progress": 0,
    "drift_counter": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(__file__)

DATA_FILE = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rf_models.pkl")
ANOMALY_PATH = os.path.join(BASE_DIR, "anomaly_model.pkl")
BASELINE_PATH = os.path.join(BASE_DIR, "baseline_stats.pkl")
LOG_PATH = os.path.join(BASE_DIR, "live_log.csv")
ADAPTIVE_MODEL_PATH = os.path.join(BASE_DIR, "rf_models_adaptive.pkl")

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

pulse = int(time.time()) % 2
if pulse == 0:
    st.markdown("🟢 System Active")
else:
    st.markdown("⚫ Monitoring...")
    
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
    margin-bottom: 25px;   /* 👈 adds vertical spacing */

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
    font-size: 24px;
    font-weight: bold;
    
}

</style>
""", unsafe_allow_html=True)


st.title(" ONGC – Flow Meter Predictive Maintenance Dashboard")


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
# RETRAINING
# =====================================================

def retrain_from_live():

    if not os.path.exists(LOG_PATH):
        st.error("No live log available.")
        return None

    df_log = pd.read_csv(LOG_PATH)

    # Use only stable rows
    df_log = df_log[df_log["Severity"].isin(["🟢 NORMAL","🟡 LOW"])]

    # Minimum safety threshold
    if len(df_log) < 200:
        st.warning("Not enough stable live data for retraining.")
        return None

    df_train = df_log[["Pressure","Temperature","Flow"]]

    # Remove extreme outliers
    z = np.abs((df_train - df_train.mean()) / df_train.std())
    df_train = df_train[(z < 3).all(axis=1)]

    models, anomaly_model, baseline = train_models(df_train)

    joblib.dump(models, ADAPTIVE_MODEL_PATH)
    joblib.dump(anomaly_model, ANOMALY_PATH)
    joblib.dump(baseline, BASELINE_PATH)

    return models, anomaly_model, baseline

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

# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.image("ongc_logo.png", width=120)

st.sidebar.header("🔧 Flow Meter Inputs")

pressure = st.sidebar.number_input(
    "Pressure (Kg/cm2)",
    min_value=2.0,
    max_value=45.0,
    value=18.0,
    step=0.5,
    key="pressure_input"
)

temperature = st.sidebar.number_input(
    "Temperature (degC)",
    min_value=5.0,
    max_value=65.0,
    value=50.0,
    step=0.5,
    key="temperature_input"
)

run = st.sidebar.button("🔄 Fetch Live Flow")

st.sidebar.markdown("### ⏱ Live Control")
st.sidebar.markdown("### 🌊 Flow Source")

flow_mode = st.sidebar.radio(
    "Select Flow Mode",
    ["Live", "Manual"],
    horizontal=True
)
manual_flow = st.sidebar.number_input(
    "Manual Flow (m3/hr)",
    min_value=0.0,
    max_value=3000.0,
    value=500.0,
    step=10.0
)
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 5, 60, 10)

# =====================================================
# SCADA BAR
# =====================================================

def scada_bar(percent):
    blocks = int(percent / 10)
    bar = "█" * blocks + "░" * (10 - blocks)
    return f"[{bar}] {percent}%"

# =====================================================
# MODEL CONTROL
# =====================================================

st.sidebar.markdown("### 🧠 Model Control")

col1, col2 = st.sidebar.columns([1,2])

with col1:
    train_btn = st.button("🧠 Train Models")

with col2:
    train_placeholder = st.empty()
    train_placeholder.markdown(scada_bar(st.session_state.train_progress))


col3, col4 = st.sidebar.columns([1,2])

with col3:
    retrain_btn = st.button("♻️ Retrain Engine")

with col4:
    retrain_placeholder = st.empty()
    retrain_placeholder.markdown(scada_bar(st.session_state.engine_progress))

flow_placeholder = st.sidebar.empty()

# =====================================================
# INDUSTRIAL TRAINING ENGINE
# =====================================================
# =====================================================
# TRAIN MODELS
# =====================================================

if train_btn:

    for i in [20, 40, 60, 80, 100]:
        st.session_state.train_progress = i
        train_placeholder.markdown(scada_bar(i))
        time.sleep(0.4)

    models, anomaly_model, baseline = train_models(load_data())

        # Reset drift memory
    st.session_state.residual_history = []
    st.session_state.drift_counter = 0

    # Optional: reset drift flag if you use one
    st.session_state.drift_active = False

    st.session_state.models = models
    st.session_state.anomaly_model = anomaly_model
    st.session_state.baseline = baseline

    st.sidebar.markdown(
        "<span style='color:#ff69b4;font-weight:bold;'>"
        "✅ Predictive Engine Updated</span>",
        unsafe_allow_html=True
    )

    time.sleep(0.8)

    st.session_state.train_progress = 0
    train_placeholder.markdown(scada_bar(0))


# =====================================================
# RETRAIN ENGINE
# =====================================================

if retrain_btn:

    for i in [25, 50, 75, 100]:
        st.session_state.engine_progress = i
        retrain_placeholder.markdown(scada_bar(i))
        time.sleep(0.4)

    result = retrain_from_live()

    if result is not None:
        models, anomaly_model, baseline = result

        st.session_state.models = models
        st.session_state.anomaly_model = anomaly_model
        st.session_state.baseline = baseline

        st.success("♻️ Adaptive Model Updated from Live Data")

    st.session_state.engine_progress = 0
    retrain_placeholder.markdown(scada_bar(0))

# =====================================================
# MAIN EXECUTION
# =====================================================

execute = run or auto_refresh

if execute:

    
    if flow_mode == "Live":
        flow = fetch_live_flow()
        flow_placeholder.success(f"Live Flow: {flow}")

    else:
        flow = manual_flow
        flow_placeholder.info(f"Manual Flow: {flow}")


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

    max_score = 0.3  # tune based on training distribution
    failure_prob = round(min(100, abs(score)/max_score * 100), 2)

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

    baseline_std = max(1e-6, np.mean(list(baseline["std"].values())))
    sensitivity = 18
    health_score = max(
        0,
        min(100, 100 - (weighted_res/baseline_std)*sensitivity)
    )

    # ======================
    # THRESHOLD-BASED RUL
    # ======================
    st.session_state.residual_history.append(weighted_res)

    MAX_HISTORY = 200
    st.session_state.residual_history = \
    st.session_state.residual_history[-MAX_HISTORY:]
    rul = "Stable"

    if len(st.session_state.residual_history) > 4:
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

    if drift_detected:
        st.session_state.drift_counter += 1
    else:
        st.session_state.drift_counter = max(
        0,
        st.session_state.drift_counter - 1
    )

    model_stability = max(0, 100 - st.session_state.drift_counter * 10)

    # 🔹 Only show status — DO NOT retrain
    if st.session_state.drift_counter >= 3:
        st.error("🚨 Persistent Drift Detected")
        st.info("Recommendation: Run Adaptive Retraining")

    elif drift_detected:
        st.warning("⚠️ Data Drift Detected – Monitoring")

    else:
        st.success("✅ System Stable")


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

r = None

if st.session_state.last_result is not None:

    r = st.session_state.last_result

    st.markdown("## 🚀 System Overview")

if r is not None:

# -----------------------------
# PRESSURE LIMIT LOGIC
# -----------------------------  
    pressure_value = r["Pressure"]

    pressure_status = "NORMAL"
    pressure_class = "normal"

    if pressure_value < 5:
        pressure_status = "BAD"
        pressure_class = "high"     # red border
    elif pressure_value < 10:
        pressure_status = "LOW"
        pressure_class = "medium"   # orange border
    elif pressure_value > 35:
        pressure_status = "HIGH"
        pressure_class = "low"      # yellow border

# -----------------------------
# TEMPERATURE LIMIT LOGIC
# -----------------------------
    temperature_value = r["Temperature"]

    temperature_status = "NORMAL"
    temperature_class = "normal"

    if temperature_value < 18:
        temperature_status = "BAD"
        temperature_class = "high"     # red border
    elif temperature_value < 24:
        temperature_status = "LOW"
        temperature_class = "medium"   # orange border
    elif temperature_value > 55:
        temperature_status = "HIGH"
        temperature_class = "low"      # yellow border

# -----------------------------
# FLOW LIMIT LOGIC
# -----------------------------

    flow_value = r["Flow"]

    flow_status = "NORMAL"
    flow_class = "normal"

    if flow_value < 100:
        flow_status = "LOW"
        flow_class = "medium"
    elif flow_value > 2400:
        flow_status = "HIGH"
        flow_class = "low"


    severity_class = "normal"
    if "LOW" in r["Severity"]:
        severity_class = "low"
    elif "MEDIUM" in r["Severity"]:
        severity_class = "medium"
    elif "HIGH" in r["Severity"]:
        severity_class = "high"

    # -------- ROW 1 --------
    col1, col2, col3 = st.columns(3, gap="large")

    # 🔹 Pressure
    with col1:
        st.markdown(f"""
        <div class="kpi-card glow {pressure_class}">
            <div class="kpi-title">Pressure</div>
            <div class="kpi-value">{round(r["Pressure"],2)} bar</div>
            <div>Δ {round(r["Pressure"]-r["P_Pred"],2)}</div>
            <div style="margin-top:8px; font-weight:bold;">
                Status: {pressure_status}
            </div>
            <div style="font-size:12px; opacity:0.6;">
                Limits: 2 – 35 bar
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 🔹 Temperature
    with col2:   # ← changed from col1 to col2
        st.markdown(f"""
        <div class="kpi-card glow {temperature_class}">
            <div class="kpi-title">Temperature</div>
            <div class="kpi-value">{round(r["Temperature"],2)} °C</div>
            <div>Δ {round(r["Temperature"]-r["T_Pred"],2)}</div>
            <div style="margin-top:8px; font-weight:bold;">
                Status: {temperature_status}
            </div>
            <div style="font-size:12px; opacity:0.6;">
                Limits: 5 – 60 °C
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 🔹 Flow
    with col3:
        st.markdown(f"""
        <div class="kpi-card glow {flow_class}">
            <div class="kpi-title">Flow</div>
            <div class="kpi-value">{round(flow_value,4)} m3/hr</div>
            <div>Δ {round(flow_value - r["F_Pred"],4)}</div>
            <div style="margin-top:8px; font-weight:bold;">
                Status: {flow_status}
            </div>
            <div style="font-size:12px; opacity:0.6;">
                Limits: 50 – 2550 m3/hr
            </div>
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


    # -------- ROW 3 --------
    colD, colE, colF = st.columns(3)

    with colD:
        st.markdown(f"""
    <div class="kpi-card glow">
        <div class="kpi-title">Failure Probability</div>
        <div class="kpi-value">{r["FailureProb"]}%</div>
    </div>
    """, unsafe_allow_html=True)

    with colE:
        st.markdown(f"""
    <div class="kpi-card glow">
        <div class="kpi-title">Maintenance Action</div>
        <div class="kpi-value">{r["Action"]}</div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# Trend Visualization
# =============================

st.markdown("## 📊 Trend Analytics")

# =====================================================
# TABS: TREND + ALERTS
# =====================================================

tab1, tab2 = st.tabs(["📊 Trend Analytics", "🚨 Recent Alerts"])

# =====================================================
# 📊 TREND TAB
# =====================================================
with tab1:

    if os.path.exists(LOG_PATH):

        df_log = pd.read_csv(LOG_PATH)
        df_log["Time"] = pd.to_datetime(df_log["Time"])
        df_log = df_log.sort_values("Time")

        # Create Error column safely
        if "Flow" in df_log.columns and "F_Pred" in df_log.columns:
            df_log["Error"] = df_log["Flow"] - df_log["F_Pred"]

        time_filter = st.selectbox(
            "Select Time Window",
            ["Last 10 Records", "Last 1 Hour", "Last 24 Hours", "Full History"]
        )

        now = datetime.now()

        if time_filter == "Last 10 Records":
            df_log = df_log.tail(10)

        elif time_filter == "Last 1 Hour":
            df_log = df_log[df_log["Time"] > now - pd.Timedelta(hours=1)]

        elif time_filter == "Last 24 Hours":
            df_log = df_log[df_log["Time"] > now - pd.Timedelta(hours=24)]

        metrics = st.multiselect(
            "Select Metrics to Plot",
            ["Flow", "Pressure", "Temperature",
            "P_Pred", "T_Pred", "F_Pred",
            "Error", "HealthScore", "FailureProb"],
            default=["Flow", "F_Pred"]
        )

        fig = go.Figure()

        for metric in metrics:
            if metric in df_log.columns:
                fig.add_trace(go.Scatter(
                    x=df_log["Time"],
                    y=df_log[metric],
                    mode="lines",
                    name=metric
                ))

        fig.update_layout(
            height=500,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white"
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        # Health Trend Indicator
        if "HealthScore" in df_log.columns and len(df_log) > 5:

            slope = np.polyfit(
                range(len(df_log)),
                df_log["HealthScore"],
                1
            )[0]

            if slope < 0:
                st.error("📉 Health Deteriorating")
            elif slope > 0:
                st.success("📈 Health Improving")
            else:
                st.info("➡ Stable Trend")

    else:
        st.warning("No log data available yet.")


# =====================================================
# 🚨 ALERT TAB
# =====================================================
with tab2:

    st.markdown("## 🚨 Critical & Medium Alerts")

    if os.path.exists(LOG_PATH):

        df_log = pd.read_csv(LOG_PATH)

        alerts = df_log[
            df_log["Severity"].isin(["🔴 HIGH", "🟠 MEDIUM", "🟡 LOW"])

        ].sort_values("Time", ascending=False)

        if len(alerts) > 0:

            st.metric("Total Alerts", len(alerts))

            st.dataframe(
                alerts.tail(50),
                use_container_width=True
            )

        else:
            st.success("✅ No critical alerts recorded")

    else:
        st.warning("No alert data available yet.")


        st.markdown("### 📊 Flow vs Predicted Flow")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=df_log["Time"],
            y=df_log["Flow"],
            mode="lines",
            name="Actual Flow"
        ))

        fig2.add_trace(go.Scatter(
            x=df_log["Time"],
            y=df_log["F_Pred"],
            mode="lines",
            name="Predicted Flow"
        ))

        fig2.update_layout(
            height=400,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white"
        )

        st.plotly_chart(fig2, use_container_width=True)
# =====================================================
# AUTO REFRESH LOOP
# =====================================================
if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

    st.markdown("### 🟢 Live Engine Status")

