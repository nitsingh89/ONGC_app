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

# =====================================================
# PATH CONFIG (RELATIVE FOR GITHUB)
# =====================================================
DATA_FILE = "data.csv"
MODEL_PATH = "rf_models.pkl"
ANOMALY_PATH = "anomaly_model.pkl"
BASELINE_PATH = "baseline_stats.pkl"
LOG_PATH = "live_log.csv"

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
# SAFE DATA LOADER
# =====================================================
def load_data():
    if not os.path.exists(DATA_FILE):
        # create dummy data so app runs on first deploy
        df = pd.DataFrame({
            "Pressure": np.random.normal(18, 1, 200),
            "Temperature": np.random.normal(60, 2, 200),
            "Flow": np.random.normal(100, 5, 200)
        })
        return df

    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = ["Pressure","Temperature","Flow"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df

# =====================================================
# FETCH LIVE FLOW
# =====================================================
def fetch_live_flow():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(LIVE_URL, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for cls in LIVE_CLASSES:
            tag = soup.find("span", class_=cls)
            if tag:
                return float(tag.text.replace(",", "").strip())
    except:
        return np.random.uniform(95,105)  # fallback

# =====================================================
# TRAIN MODELS
# =====================================================
def train_models(df):

    models = {
        "pressure": RandomForestRegressor(n_estimators=200, random_state=42),
        "temperature": RandomForestRegressor(n_estimators=200, random_state=42),
        "flow": RandomForestRegressor(n_estimators=200, random_state=42)
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

# =====================================================
# LOAD OR TRAIN
# =====================================================
@st.cache_resource
def load_or_train():
    df = load_data()
    if not all(map(os.path.exists, [MODEL_PATH, ANOMALY_PATH, BASELINE_PATH])):
        return train_models(df)
    return (
        joblib.load(MODEL_PATH),
        joblib.load(ANOMALY_PATH),
        joblib.load(BASELINE_PATH)
    )

models, anomaly_model, baseline = load_or_train()
st.success("✅ Models Ready")

# =====================================================
# SIDEBAR INPUT
# =====================================================
if os.path.exists("ongc_logo.png"):
    st.sidebar.image("ongc_logo.png", width=80)

pressure = st.sidebar.number_input("Pressure", value=18.0)
temperature = st.sidebar.number_input("Temperature", value=60.0)

run = st.sidebar.button("Fetch Live Flow")

# =====================================================
# MAIN EXECUTION
# =====================================================
if run:

    flow = fetch_live_flow()
    st.sidebar.success(f"Live Flow: {flow:.2f}")

    p_pred = models["pressure"].predict([[temperature, flow]])[0]
    t_pred = models["temperature"].predict([[pressure, flow]])[0]
    f_pred = models["flow"].predict([[pressure, temperature]])[0]

    p_res = pressure - p_pred
    t_res = temperature - t_pred
    f_res = flow - f_pred

    residuals = pd.DataFrame([[p_res,t_res,f_res]],
                             columns=["P_res","T_res","F_res"])

    score = anomaly_model.decision_function(residuals)[0]
    flag = anomaly_model.predict(residuals)[0]

    severity = "NORMAL"
    if flag == -1:
        severity = "LOW"
        if score < -0.12: severity = "MEDIUM"
        if score < -0.25: severity = "HIGH"

    health_score = max(0, min(100, 100 - np.mean(np.abs([p_res,t_res,f_res]))*5))

    st.metric("Flow", f"{flow:.2f}")
    st.metric("Health Score", f"{health_score:.1f}%")
    st.metric("Anomaly", severity)

    # LOG
    log = pd.DataFrame([[datetime.now(),pressure,temperature,flow,
                         p_pred,t_pred,f_pred,
                         score,severity,health_score]],
        columns=["Time","Pressure","Temperature","Flow",
                 "P_Pred","T_Pred","F_Pred",
                 "Score","Severity","HealthScore"])

    log.to_csv(LOG_PATH, mode='a', header=not os.path.exists(LOG_PATH), index=False)

# =====================================================
# TREND CHART
# =====================================================
if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df_log["Flow"], mode='lines', name="Actual Flow"))
    fig.add_trace(go.Scatter(y=df_log["F_Pred"], mode='lines', name="Predicted Flow"))

    st.plotly_chart(fig, use_container_width=True)
