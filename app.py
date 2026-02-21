import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =====================================================
# PATH CONFIG
# =====================================================
DATA_FILE = "data.csv"
MODEL_PATH = "rf_models.pkl"
ANOMALY_PATH = "anomaly_model.pkl"
BASELINE_PATH = "baseline_stats.pkl"
LOG_PATH = "live_log.csv"

LOG_COLUMNS = [
    "Time","Pressure","Temperature","Flow",
    "P_Pred","T_Pred","F_Pred",
    "Score","Severity","FailureProb",
    "HealthScore","RUL","Action","ModelStability"
]

LIVE_URL = "https://markets.businessinsider.com/currencies/eth-usd"
LIVE_CLASSES = [
    "price-section__current-value",
    "price-section__absolute-value"
]

DRIFT_Z_THRESHOLD = 2.5

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="ONGC Predictive Maintenance", layout="wide")
st.title("ONGC – Flow Meter Predictive Maintenance Dashboard")

# =====================================================
# SESSION STATE INIT
# =====================================================
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("residual_history", [])
st.session_state.setdefault("drift_counter", 0)

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
        "std": df[["P_res","T_res","F_res"]].std().replace(0,1e-6).to_dict()
    }

    joblib.dump(models, MODEL_PATH)
    joblib.dump(iso, ANOMALY_PATH)
    joblib.dump(baseline, BASELINE_PATH)

    return models, iso, baseline

def detect_drift(residuals, baseline):
    for col in residuals.columns:
        z = abs((residuals[col].iloc[0] - baseline["mean"][col]) /
                baseline["std"][col])
        if z > DRIFT_Z_THRESHOLD:
            return True
    return False

# =====================================================
# LOAD OR TRAIN
# =====================================================
@st.cache_resource
def load_or_train():
    df = load_data()
    if not all(map(os.path.exists,[MODEL_PATH,ANOMALY_PATH,BASELINE_PATH])):
        return train_models(df)
    return (
        joblib.load(MODEL_PATH),
        joblib.load(ANOMALY_PATH),
        joblib.load(BASELINE_PATH)
    )

models, anomaly_model, baseline = load_or_train()
st.success("✅ Models Ready")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Flow Inputs")
pressure = st.sidebar.number_input("Pressure", value=18.0)
temperature = st.sidebar.number_input("Temperature", value=60.0)

run = st.sidebar.button("Fetch Live Flow")

st.sidebar.markdown("### Auto Refresh")
enable_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Interval (sec)",5,60,10)

if enable_refresh:
    st_autorefresh(interval=refresh_interval*1000, key="refresh")

# =====================================================
# MAIN EXECUTION (ONLY ONE BLOCK)
# =====================================================
# =====================================================
# MAIN EXECUTION ENGINE (Button + Auto Refresh)
# =====================================================

execute = run or auto_refresh

if execute:

    try:
        flow = fetch_live_flow()
        st.sidebar.success(f"Live Flow: {flow}")
    except:
        st.warning("Live flow fetch failed")
        flow = st.session_state.last_flow

    if flow is not None:

        st.session_state.last_flow = flow

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
        # RUL ESTIMATION
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
        # DRIFT CONTROL
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
        # MAINTENANCE ACTION
        # ======================
        if severity == "🔴 HIGH":
            action = "Immediate Shutdown Recommended"
        elif severity == "🟠 MEDIUM":
            action = "Inspect within 24 hrs"
        elif severity == "🟡 LOW":
            action = "Monitor Closely"
        else:
            action = "Normal Operation"

        if severity in ["🔴 HIGH","🟠 MEDIUM"]:
            st.error(f"🚨 ALERT: {action}")

        # ======================
        # SAFE LOGGING
        # ======================
        LOG_COLUMNS = ["Time","Pressure","Temperature","Flow",
                       "P_Pred","T_Pred","F_Pred",
                       "Score","Severity","FailureProb",
                       "HealthScore","RUL","Action","ModelStability"]

        log = pd.DataFrame([[datetime.now(),pressure,temperature,flow,
                             p_pred,t_pred,f_pred,
                             score,severity,failure_prob,
                             health_score,rul,action,model_stability]],
                           columns=LOG_COLUMNS)

        log.to_csv(LOG_PATH, mode='a',
                   header=not os.path.exists(LOG_PATH),
                   index=False)

        st.session_state.last_result = log.iloc[0]


# =====================================================
# DISPLAY
# =====================================================
if st.session_state.last_result is not None:

    r = st.session_state.last_result
    st.metric("Pressure", round(r["Pressure"],2))
    st.metric("Temperature", round(r["Temperature"],2))
    st.metric("Flow", round(r["Flow"],4))
    st.metric("Health Score", round(r["HealthScore"],2))
    st.metric("Failure Probability", f"{r['FailureProb']}%")
    st.metric("Maintenance Action", r["Action"])

# =====================================================
# TREND GRAPH (SAFE READ)
# =====================================================
if os.path.exists(LOG_PATH):

    try:
        df_log = pd.read_csv(LOG_PATH)
    except:
        st.warning("Log file corrupted. Resetting log.")
        os.remove(LOG_PATH)
        df_log = pd.DataFrame()

    if not df_log.empty:

        df_log["Error"] = df_log["Flow"] - df_log["F_Pred"]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=df_log["Flow"],
            mode='lines',
            name="Actual Flow"
        ))

        fig.add_trace(go.Scatter(
            y=df_log["F_Pred"],
            mode='lines',
            name="Predicted Flow",
            line=dict(dash='dash')
        ))

        fig.add_trace(go.Scatter(
            y=df_log["Error"],
            mode='lines',
            name="Prediction Error",
            yaxis="y2"
        ))

        fig.update_layout(
            yaxis=dict(title="Flow"),
            yaxis2=dict(title="Error",overlaying="y",side="right")
        )

        st.plotly_chart(fig,use_container_width=True)
# =====================================================
# AUTO REFRESH LOOP
# =====================================================
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

