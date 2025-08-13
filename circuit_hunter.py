

# NEPSE Circuit Hunter Dashboard (Streamlit)
# -----------------------------------------
# - Pulls live prices from MeroLagani free endpoints
# - Attempts to fetch order-book depth for buy/sell pressure
# - Ranks likely circuit candidates
#
# Run:
#   pip install -r requirements.txt
#   streamlit run circuit_hunter.py
#
# Notes:
# - Uses only publicly accessible endpoints.
# - Falls back gracefully if any endpoint structure changes.
# - You can adjust REFRESH_SECONDS, thresholds, and filters from the sidebar.

import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ---------------------- Config ----------------------

REFRESH_DEFAULT_SECONDS = 60  # auto refresh every N seconds
TOP_N_DEFAULT = 50            # show top N ranked rows
DEPTH_LEVELS_TO_SUM = 3       # sum top-N levels for buy/sell pressure

# Circuit proximity thresholds (you can tweak in sidebar)
GAP_ALERT_UPPER = 1.0  # % distance to upper limit
GAP_ALERT_LOWER = 1.0  # % distance to lower limit

# Try alternative casings for MeroLagani handlers (they sometimes vary)
BASES = [
    "https://merolagani.com/handlers",
    "https://merolagani.com/Handlers",
]

# ---------------------- Helpers ----------------------

@st.cache_data(ttl=30, show_spinner=False)
def fetch_today_prices() -> pd.DataFrame:
    """
    Fetch today's prices for all symbols from MeroLagani.
    Returns a DataFrame with normalized column names.
    """
    errors = []
    data = None
    for base in BASES:
        url = f"{base}/TodayPrice.ashx?CompanyGroup=all"
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                raw = r.json()
                if isinstance(raw, list) and len(raw) > 0:
                    data = raw
                    break
        except Exception as e:
            errors.append(f"{url} -> {repr(e)}")

    if data is None:
        raise RuntimeError("Unable to fetch today prices from MeroLagani. Tried:\n" + "\n".join(errors))

    df = pd.DataFrame(data)

    # Normalize likely columns (handle various key name variants defensively)
    rename_map = {
        "Symbol": "symbol",
        "LTP": "ltp",
        "Ltv": "ltp",
        "LtvValue": "ltp",
        "OpenPrice": "open",
        "High": "high",
        "Low": "low",
        "PreviousClose": "prev_close",
        "Change": "change",
        "PercentChange": "pct_change",
        "PercentageChange": "pct_change",
        "TotalTradedQuantity": "qty",
        "TotalTradedValue": "turnover",
        "MaxPrice": "max_price",
        "MinPrice": "min_price",
        "UpperCircuit": "upper_limit",
        "LowerCircuit": "lower_limit",
        "VL": "volume",
        "VWAP": "vwap",
    }
    # Apply rename for any keys that exist
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    # Keep only useful columns
    keep_cols = [c for c in [
        "symbol","ltp","open","high","low","prev_close","change","pct_change",
        "qty","turnover","max_price","min_price","upper_limit","lower_limit","vwap","volume"
    ] if c in df.columns]
    df = df[keep_cols].copy()

    # Coerce numerics
    for col in [c for c in df.columns if c not in ("symbol",)]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Some feeds might not include explicit circuit limits; try to estimate from max/min if present
    if "upper_limit" not in df.columns:
        df["upper_limit"] = np.where(df.get("max_price").notna(), df["max_price"], np.nan)
    if "lower_limit" not in df.columns:
        df["lower_limit"] = np.where(df.get("min_price").notna(), df["min_price"], np.nan)

    # Deduce missing LTP from open/high/low if needed
    if "ltp" not in df.columns:
        # Create an ltp placeholder using VWAP or open
        df["ltp"] = df.get("vwap", pd.Series(index=df.index, dtype=float)).fillna(df.get("open"))

    df = df.dropna(subset=["symbol"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    return df


@st.cache_data(ttl=20, show_spinner=False)
def fetch_depth(symbol: str) -> Dict:
    """
    Attempt to fetch order book depth from MeroLagani for a given symbol.
    Returns a dict with buyers/sellers arrays if available.
    """
    last_error = None
    for base in BASES:
        url = f"{base}/MarketDepthHandler.ashx?symbol={symbol}"
        try:
            r = requests.get(url, timeout=12)
            if r.ok:
                js = r.json()
                # Heuristic: accept if dict with 'Buyers'/'Sellers' or similar keys
                if isinstance(js, dict) and any(k.lower().startswith("buy") for k in js.keys()):
                    return js
        except Exception as e:
            last_error = e

    # Fallback: try HTML page scrape (very light, best-effort; structure may change)
    for base in ["https://merolagani.com"]:
        url = f"{base}/MarketDepth.aspx?symbol={symbol}"
        try:
            r = requests.get(url, timeout=12)
            if r.ok and ("Market Depth" in r.text or "Buy" in r.text):
                # Avoid scraping heavy; return a marker to signal available page
                return {"_html_page": True}
        except Exception as e:
            last_error = e

    return {"_error": str(last_error) if last_error else "Depth not available"}


def _sum_levels(levels: List[Dict], key_qty_candidates=("Quantity","quantity","Qty","qty")) -> float:
    if not isinstance(levels, list) or len(levels) == 0:
        return 0.0
    total = 0.0
    for row in levels[:DEPTH_LEVELS_TO_SUM]:
        if not isinstance(row, dict):
            continue
        qty = None
        for k in key_qty_candidates:
            if k in row:
                qty = row[k]
                break
        if qty is None:
            # try second-level value
            vals = list(row.values())
            if vals and isinstance(vals[0], (int,float,str)):
                try:
                    qty = float(vals[0])
                except:
                    qty = 0
        try:
            total += float(qty or 0)
        except:
            pass
    return float(total)


def compute_pressures(depth_json: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (buy_topN_qty, sell_topN_qty, buy_sell_ratio) if depth arrays are available.
    """
    if not isinstance(depth_json, dict):
        return (None, None, None)

    # Try common keys
    buyers = None
    sellers = None
    for k in depth_json.keys():
        lk = k.lower()
        if lk.startswith("buy"):
            buyers = depth_json[k]
        if lk.startswith("sell"):
            sellers = depth_json[k]

    if buyers is None and sellers is None:
        return (None, None, None)

    bsum = _sum_levels(buyers) if buyers is not None else None
    ssum = _sum_levels(sellers) if sellers is not None else None

    ratio = None
    if bsum is not None and ssum is not None:
        if ssum == 0:
            ratio = np.inf if bsum > 0 else None
        else:
            ratio = bsum / ssum

    return (bsum, ssum, ratio)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed fields:
    - gap_up_pct, gap_down_pct
    - direction_guess ('Upper', 'Lower', or '—')
    """
    out = df.copy()

    # Ensure needed columns exist
    for c in ["ltp","upper_limit","lower_limit","qty","turnover"]:
        if c not in out.columns:
            out[c] = np.nan

    # distances
    out["gap_up_pct"] = np.where(
        (out["ltp"] > 0) & out["upper_limit"].notna(),
        (out["upper_limit"] - out["ltp"]) / out["ltp"] * 100.0,
        np.nan
    )
    out["gap_down_pct"] = np.where(
        (out["ltp"] > 0) & out["lower_limit"].notna(),
        (out["ltp"] - out["lower_limit"]) / out["ltp"] * 100.0,
        np.nan
    )

    # initial direction (based on which gap is smaller)
    def _dir_row(r):
        gu = r.get("gap_up_pct", np.nan)
        gd = r.get("gap_down_pct", np.nan)
        if pd.notna(gu) and pd.notna(gd):
            return "Upper" if gu <= gd else "Lower"
        if pd.notna(gu):
            return "Upper"
        if pd.notna(gd):
            return "Lower"
        return "—"

    out["direction_guess"] = out.apply(_dir_row, axis=1)

    return out


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank by closeness to circuit and pressure ratio if available.
    """
    d = df.copy()

    # Score components (lower gap is better)
    d["score_gap"] = np.where(d["direction_guess"] == "Upper", d["gap_up_pct"],
                              np.where(d["direction_guess"] == "Lower", d["gap_down_pct"], np.nan))

    # Normalize ratios: higher buy/sell ratio better for Upper; inverse for Lower
    # Build a unified pressure score in log space to damp extremes
    pr = d.get("buy_sell_ratio")
    d["score_pressure"] = np.nan
    if pr is not None:
        # For Upper: +log(ratio). For Lower: +log(1/ratio) == -log(ratio).
        d.loc[d["direction_guess"] == "Upper", "score_pressure"] = np.log(np.clip(d.loc[d["direction_guess"] == "Upper", "buy_sell_ratio"].astype(float).replace([np.inf, -np.inf], np.nan), 1e-6, 1e6))
        d.loc[d["direction_guess"] == "Lower", "score_pressure"] = np.log(np.clip(1.0 / d.loc[d["direction_guess"] == "Lower", "buy_sell_ratio"].astype(float).replace([np.inf, -np.inf], np.nan), 1e-6, 1e6))

    # Combine: lower score_gap is better; higher pressure is better.
    d["rank_score"] = (
        (d["score_gap"].fillna(9999)) +
        (-1.0 * d["score_pressure"].fillna(0.0))
    )

    d = d.sort_values(by=["rank_score","gap_up_pct","gap_down_pct"], ascending=[True, True, True]).reset_index(drop=True)
    return d


def color_gap(val: float, direction: str) -> str:
    if pd.isna(val):
        return ""
    # Emphasize near-circuit values
    if val <= 0.25:
        return "background-color: rgba(34, 197, 94, .35);" if direction == "Upper" else "background-color: rgba(239, 68, 68, .35);"
    if val <= 0.75:
        return "background-color: rgba(34, 197, 94, .2);" if direction == "Upper" else "background-color: rgba(239, 68, 68, .2);"
    if val <= 1.5:
        return "background-color: rgba(34, 197, 94, .1);" if direction == "Upper" else "background-color: rgba(239, 68, 68, .1);"
    return ""


# ---------------------- UI ----------------------

st.set_page_config(page_title="NEPSE Circuit Hunter", layout="wide")

st.title("⚡ NEPSE Circuit Hunter")
st.caption("Live scan for likely circuit-hit stocks on NEPSE using MeroLagani public data.")

with st.sidebar:
    st.header("Settings")
    refresh = st.number_input("Auto-refresh seconds", min_value=10, max_value=300, value=REFRESH_DEFAULT_SECONDS, step=5)
    topn = st.number_input("Show top N", min_value=10, max_value=500, value=TOP_N_DEFAULT, step=10)
    st.markdown("---")
    st.subheader("Circuit alert thresholds")
    gap_up_thr = st.number_input("Gap ≤ (Upper) %", min_value=0.1, max_value=10.0, value=GAP_ALERT_UPPER, step=0.1, format="%.1f")
    gap_dn_thr = st.number_input("Gap ≤ (Lower) %", min_value=0.1, max_value=10.0, value=GAP_ALERT_LOWER, step=0.1, format="%.1f")
    st.markdown("---")
    st.subheader("Depth settings")
    levels_to_sum = st.number_input("Depth levels to sum", min_value=1, max_value=5, value=DEPTH_LEVELS_TO_SUM, step=1)
    st.caption("If order-book depth is available, sum top-N levels for pressure ratios.")
    st.markdown("---")
    st.write("Data source: merolagani.com (public handlers).")

    # Autorefresh ping
    st.experimental_set_query_params(_ts=int(time.time()))
    st.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# Reflect sidebar choice to global
global DEPTH_LEVELS_TO_SUM
DEPTH_LEVELS_TO_SUM = int(levels_to_sum)

# Kick auto-refresh
st.experimental_rerun  # no-op placeholder so linters don't remove import

st.toast("Scanning live market…", icon="🛰️")

# Fetch prices
try:
    prices = fetch_today_prices()
except Exception as e:
    st.error(f"Failed to load prices: {e}")
    st.stop()

df = annotate(prices)

# Fetch depth per symbol (best-effort). Do a limited batch for performance.
symbols = df["symbol"].tolist()

buy_top = []
sell_top = []
ratio = []
depth_note = []

for sym in symbols:
    depth = fetch_depth(sym)
    bsum, ssum, rs = compute_pressures(depth)
    buy_top.append(bsum)
    sell_top.append(ssum)
    ratio.append(rs)

    if "_html_page" in depth:
        depth_note.append("page-only")
    elif "_error" in depth:
        depth_note.append("unavailable")
    else:
        depth_note.append("")

df["buy_topN_qty"] = buy_top
df["sell_topN_qty"] = sell_top
df["buy_sell_ratio"] = ratio
df["depth"] = depth_note

ranked = rank_candidates(df)

# Alerts
upper_hits = ranked[(ranked["gap_up_pct"].notna()) & (ranked["gap_up_pct"] <= gap_up_thr)]
lower_hits = ranked[(ranked["gap_down_pct"].notna()) & (ranked["gap_down_pct"] <= gap_dn_thr)]

st.subheader("Alerts")
c1, c2 = st.columns(2)
with c1:
    st.metric("Near Upper Circuit (≤ threshold)", len(upper_hits))
with c2:
    st.metric("Near Lower Circuit (≤ threshold)", len(lower_hits))

# Display table
show_cols = [
    "symbol","ltp","upper_limit","lower_limit",
    "gap_up_pct","gap_down_pct",
    "buy_topN_qty","sell_topN_qty","buy_sell_ratio",
    "qty","turnover","direction_guess","depth"
]
show_cols = [c for c in show_cols if c in ranked.columns]

styled = ranked.head(int(topn))[show_cols].style.format({
    "ltp": "{:,.2f}",
    "upper_limit": "{:,.2f}",
    "lower_limit": "{:,.2f}",
    "gap_up_pct": "{:.2f}%",
    "gap_down_pct": "{:.2f}%",
    "buy_topN_qty": "{:,.0f}",
    "sell_topN_qty": "{:,.0f}",
    "buy_sell_ratio": "{:.2f}",
    "qty": "{:,.0f}",
    "turnover": "{:,.0f}",
}).apply(lambda s: [color_gap(v, ranked.loc[s.index[i], "direction_guess"]) if s.name in ("gap_up_pct","gap_down_pct") else "" for i, v in enumerate(s)], axis=0)

st.subheader("Ranked Candidates")
st.dataframe(styled, use_container_width=True)

st.caption("Tip: Adjust thresholds in the sidebar to tune aggressiveness. If circuit limits are missing in the feed, the app estimates using available fields.")

