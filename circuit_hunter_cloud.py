
import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime

# ---------------------- CONFIG ----------------------
REFRESH_SEC = 60
NEPSE_API = "https://www.nepalstock.com.np/api/nots/nepse-data/today-price"  # Live price API

# ---------------------- FUNCTIONS ----------------------
@st.cache_data(ttl=30)
def fetch_nepse_data():
    """Fetch today's market data from NEPSE API"""
    try:
        resp = requests.get(NEPSE_API, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json()
        if "data" not in data:
            raise ValueError("Invalid response format from NEPSE")
        df = pd.DataFrame(data["data"])
        # Clean and normalize
        df.rename(columns={
            "symbol": "Symbol",
            "lastTradedPrice": "LTP",
            "maxPrice": "Upper",
            "minPrice": "Lower",
            "totalTradedQuantity": "Volume",
            "totalTradedValue": "Turnover"
        }, inplace=True, errors="ignore")
        return df
    except Exception as e:
        st.error(f"Error fetching NEPSE data: {e}")
        return pd.DataFrame()

def compute_metrics(df):
    """Add gap to circuit and estimated buy/sell pressure"""
    if df.empty:
        return df
    df["Gap_Up_%"] = ((df["Upper"] - df["LTP"]) / df["LTP"]) * 100
    df["Gap_Down_%"] = ((df["LTP"] - df["Lower"]) / df["LTP"]) * 100
    # Simple buy/sell pressure estimation
    df["Pressure"] = np.where(df["Gap_Up_%"] < df["Gap_Down_%"],
                              df["Volume"] / (df["Turnover"] / df["LTP"] + 1),
                              -(df["Volume"] / (df["Turnover"] / df["LTP"] + 1)))
    return df

def rank_candidates(df, gap_threshold=1.0):
    """Rank likely circuit candidates"""
    upper_hits = df[df["Gap_Up_%"] <= gap_threshold].copy()
    lower_hits = df[df["Gap_Down_%"] <= gap_threshold].copy()
    upper_hits["Direction"] = "Upper"
    lower_hits["Direction"] = "Lower"
    return pd.concat([upper_hits, lower_hits], ignore_index=True).sort_values("Gap_Up_%")

# ---------------------- UI ----------------------
st.set_page_config(page_title="NEPSE Circuit Hunter", layout="wide")
st.title("⚡ NEPSE Circuit Hunter (Cloud-Safe)")
st.caption("Live scan for likely circuit-hit stocks from NEPSE's own API. Refreshes every 60 seconds.")

# Auto-refresh
st_autorefresh = st.experimental_rerun

# Sidebar settings
gap_thr = st.sidebar.number_input("Gap Threshold %", 0.1, 5.0, 1.0, step=0.1)
top_n = st.sidebar.number_input("Show Top N", 5, 50, 20, step=1)

# Fetch + process
df = fetch_nepse_data()
if not df.empty:
    df = compute_metrics(df)
    ranked = rank_candidates(df, gap_thr)
    st.subheader("Likely Circuit Candidates")
    st.dataframe(ranked.head(top_n)[["Symbol", "LTP", "Upper", "Lower", "Gap_Up_%", "Gap_Down_%", "Volume", "Turnover", "Direction"]])
else:
    st.warning("No data available from NEPSE API right now.")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
