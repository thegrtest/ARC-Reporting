#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dash web dashboard for CIP Tag Poller

Reads:
    logs/raw_data_YYYY-MM-DD.csv   (latest day for live/current & live-hour avg)
    logs/hourly_averages.csv       (hourly aggregates)

Exposes a network dashboard with:
    - For EVERY tag:
        * Gauge: Current Value
        * Gauge: Last Full Hour Average
        * Gauge: Live Hourly Average (current hour so far)
    - Color-coded gauges using thresholds.json (configurable in Thresholds tab).

Run:
    python cip_dash_server.py

Then open:
    http://<this_machine_ip>:8050
"""

import os
import csv
import json
import hashlib
import getpass
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_daq as daq

# ----------------- config -----------------

LOG_DIR = "logs"
HOURLY_CSV = os.path.join(LOG_DIR, "hourly_averages.csv")
THRESHOLDS_JSON = os.path.join(LOG_DIR, "thresholds.json")
SETTINGS_JSON = "settings.json"  # optional: to discover tags before any data
CONFIG_CHANGES_CSV = os.path.join(LOG_DIR, "config_changes.csv")
CONFIG_CHANGE_HEADERS = [
    "timestamp",
    "user",
    "field",
    "old_value",
    "new_value",
    "reason",
]

REFRESH_MS = 5000  # dashboard refresh interval (ms)

os.makedirs(LOG_DIR, exist_ok=True)


# ----------------- helpers: files & thresholds -----------------


def ensure_dir(path: str) -> None:
    d = os.path.abspath(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def ensure_csv(path: str, headers: List[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def log_config_change(field: str, old_value: object, new_value: object, reason: str) -> None:
    ensure_csv(CONFIG_CHANGES_CSV, CONFIG_CHANGE_HEADERS)
    try:
        user = getpass.getuser() or "Workstation"
    except Exception:
        user = "Workstation"

    ts = datetime.now().isoformat(timespec="seconds")
    row = [
        ts,
        user,
        field,
        "" if old_value is None else str(old_value),
        "" if new_value is None else str(new_value),
        reason,
    ]
    try:
        with open(CONFIG_CHANGES_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception:
        pass


def compute_config_version_text() -> str:
    if not os.path.exists(SETTINGS_JSON):
        return "Config: not found"
    try:
        with open(SETTINGS_JSON, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        mdate = datetime.fromtimestamp(os.path.getmtime(SETTINGS_JSON)).strftime(
            "%Y-%m-%d"
        )
        return f"Config: v{mdate} ({digest[:8]})"
    except Exception:
        return "Config: unavailable"


def get_latest_raw_path() -> Optional[str]:
    """
    Return path to latest raw_data_YYYY-MM-DD.csv in LOG_DIR, or None.
    """
    if not os.path.isdir(LOG_DIR):
        return None
    candidates: List[str] = []
    for name in os.listdir(LOG_DIR):
        if name.startswith("raw_data_") and name.endswith(".csv"):
            candidates.append(name)
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(LOG_DIR, candidates[-1])


def load_latest_raw_df() -> pd.DataFrame:
    """
    Load the latest daily raw CSV as DataFrame:
        timestamp,date,time,tag,value,status
    Returns empty DataFrame on any error.
    """
    path = get_latest_raw_path()
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "date", "time", "tag", "value", "status"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "date", "time", "tag", "value", "status"])

    # Normalize columns
    for col in ["timestamp", "date", "time", "tag", "value", "status"]:
        if col not in df.columns:
            df[col] = None

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception:
        df["timestamp"] = pd.NaT

    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    df["status_str"] = df["status"].astype(str)
    df["ok"] = df["status_str"].str.lower().eq("success")

    return df


def load_hourly_stats() -> pd.DataFrame:
    """
    Load hourly averages as DataFrame with columns:
        hour_start, hour_end, tag, avg_value, sample_count
    Returns empty DataFrame on any error.
    """
    if not os.path.exists(HOURLY_CSV):
        return pd.DataFrame(
            columns=["hour_start", "hour_end", "tag", "avg_value", "sample_count"]
        )
    try:
        df = pd.read_csv(HOURLY_CSV)
    except Exception:
        return pd.DataFrame(
            columns=["hour_start", "hour_end", "tag", "avg_value", "sample_count"]
        )

    for c in ["hour_start", "hour_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_thresholds() -> Dict[str, Dict[str, float]]:
    """
    Load per-tag thresholds from JSON if available.
    thresholds.json format:
        { "TagName": {"low": float, "high": float}, ... }
    """
    if not os.path.exists(THRESHOLDS_JSON):
        return {}
    try:
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_thresholds(th: Dict[str, Dict[str, float]]) -> None:
    ensure_dir(os.path.dirname(THRESHOLDS_JSON) or ".")
    try:
        with open(THRESHOLDS_JSON, "w", encoding="utf-8") as f:
            json.dump(th, f, indent=2)
    except Exception:
        pass


def load_configured_tags_from_settings() -> List[str]:
    """
    Optionally discover tags from settings.json (the PySide app config),
    so we can show cards for tags even before they have data.
    """
    if not os.path.exists(SETTINGS_JSON):
        return []
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    tags_text = data.get("tags", "")
    if not isinstance(tags_text, str):
        return []

    tags = [line.strip() for line in tags_text.splitlines() if line.strip()]
    return tags


# ----------------- helpers: classification & ranges -----------------


def classify_value(value: float, low: float, high: float) -> str:
    """
    Return status string "good", "warning", or "bad" based on thresholds.
    - good: low <= value <= high
    - warning: within 10% outside range
    - bad: far outside
    """
    if value != value:  # NaN
        return "unknown"

    if low <= value <= high:
        return "good"

    width = max(1e-6, high - low)
    if value < low and (low - value) <= 0.1 * width:
        return "warning"
    if value > high and (value - high) <= 0.1 * width:
        return "warning"

    return "bad"


def status_color(status: str) -> str:
    if status == "good":
        return "#4caf50"  # green
    if status == "warning":
        return "#fdd835"  # yellow
    if status == "bad":
        return "#ef5350"  # red
    return "#78909c"      # gray


def compute_gauge_range(
    low: Optional[float],
    high: Optional[float],
    sample_values: List[float],
) -> Tuple[float, float, float, float]:
    """
    Compute (low, high, gauge_min, gauge_max) for a tag.
    - If thresholds exist, use them (and swap if low/high reversed).
    - Otherwise derive a reasonable range from sample values or fall back to [0, 1].
    Gauge min/max are rounded to 2 decimals to avoid messy tick labels.
    """
    values = [v for v in sample_values if v == v]

    # Thresholds present
    if low is not None and high is not None and high != low:
        if high < low:
            low, high = high, low
        center = (low + high) / 2.0
        span = max(1.0, abs(high - low) * 1.5)
        gmin = round(center - span / 2.0, 2)
        gmax = round(center + span / 2.0, 2)
        return low, high, gmin, gmax

    # No thresholds: derive from data if possible
    if values:
        vmin = min(values)
        vmax = max(values)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
    else:
        vmin = 0.0
        vmax = 1.0

    center = (vmin + vmax) / 2.0
    span = max(1.0, (vmax - vmin) * 1.5)
    gmin = round(center - span / 2.0, 2)
    gmax = round(center + span / 2.0, 2)

    return vmin, vmax, gmin, gmax


# ----------------- helpers: extraction from dataframes -----------------


def extract_raw_stats(raw_df: pd.DataFrame):
    """
    From latest-day raw_df, compute:
        - latest_by_tag: {tag: (value_num, plc_status)}
        - current_hour_avg: {tag: avg_value for current hour}
        - last_update_ts: datetime or None
    """
    if raw_df.empty or "tag" not in raw_df.columns:
        return {}, {}, None

    try:
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    except Exception:
        raw_df["timestamp"] = pd.NaT

    if "value_num" not in raw_df.columns:
        raw_df["value_num"] = pd.to_numeric(raw_df.get("value", None), errors="coerce")

    if "status_str" not in raw_df.columns:
        raw_df["status_str"] = raw_df.get("status", "").astype(str)

    raw_df["ok"] = raw_df["status_str"].str.lower().eq("success")

    # Latest per tag
    df_sorted = raw_df.sort_values("timestamp")
    latest_by_tag: Dict[str, Tuple[float, str]] = {}
    try:
        latest_rows = df_sorted.groupby("tag").tail(1)
        for _, row in latest_rows.iterrows():
            tag = str(row["tag"])
            val = float(row["value_num"]) if row["value_num"] == row["value_num"] else float("nan")
            status = str(row["status_str"])
            latest_by_tag[tag] = (val, status)
    except Exception:
        pass

    # Current hour avg per tag
    now = datetime.now()
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    hour_end = hour_start + timedelta(hours=1)
    mask = (
        (raw_df["timestamp"] >= hour_start)
        & (raw_df["timestamp"] < hour_end)
        & raw_df["ok"]
    )
    current_hour_avg: Dict[str, float] = {}
    try:
        ch_df = raw_df[mask]
        if not ch_df.empty:
            grp = ch_df.groupby("tag")["value_num"].mean()
            current_hour_avg = {str(tag): float(val) for tag, val in grp.items()}
    except Exception:
        pass

    # Global last update
    try:
        last_ts = raw_df["timestamp"].max()
        if pd.isna(last_ts):
            last_ts = None
    except Exception:
        last_ts = None

    return latest_by_tag, current_hour_avg, last_ts


def extract_last_full_hour(
    hourly_df: pd.DataFrame,
) -> Dict[str, Tuple[float, datetime, datetime, int]]:
    """
    From hourly_df, extract for each tag the last full hour:
        {tag: (avg_value, hour_start, hour_end, sample_count)}
    Any row with sample_count <= 0 is treated as "no data".
    """
    result: Dict[str, Tuple[float, datetime, datetime, int]] = {}
    if hourly_df.empty or "tag" not in hourly_df.columns:
        return result

    try:
        df = hourly_df.copy()
        df = df.sort_values("hour_start")
        for tag, group in df.groupby("tag"):
            row = group.iloc[-1]
            hs = row["hour_start"]
            he = row["hour_end"]
            try:
                cnt = int(row.get("sample_count", 0))
            except Exception:
                cnt = 0
            if cnt <= 0:
                # treat as no data
                result[str(tag)] = (float("nan"), hs, he, 0)
                continue
            try:
                avg = float(row["avg_value"])
            except Exception:
                avg = float("nan")
            result[str(tag)] = (avg, hs, he, cnt)
    except Exception:
        pass

    return result


def looks_numeric_tag(tag: str) -> bool:
    """
    Return True if the tag string looks like a pure numeric value
    (float or int) without letters. Used to filter out corrupted tags.
    """
    try:
        float(tag)
        # if it parses AND there are no alphabetic characters, treat as numeric
        return not any(ch.isalpha() for ch in tag)
    except Exception:
        return False


def discover_all_tags_for_dropdown() -> List[str]:
    """
    Discover tags from latest raw data, hourly stats, and settings.json.
    Filters out tags that look purely numeric unless they also appear
    in settings.json or raw data.
    """
    raw_df = load_latest_raw_df()
    hourly_df = load_hourly_stats()
    tags_raw = set()
    tags_hourly = set()

    try:
        if not raw_df.empty and "tag" in raw_df.columns:
            tags_raw = set(raw_df["tag"].dropna().astype(str))
    except Exception:
        pass

    try:
        if not hourly_df.empty and "tag" in hourly_df.columns:
            tags_hourly = set(hourly_df["tag"].dropna().astype(str))
    except Exception:
        pass

    tags_cfg = set(load_configured_tags_from_settings())

    # start with tags we trust: from raw data or config
    trusted = tags_raw | tags_cfg

    # add hourly-only tags that do not look numeric
    for t in tags_hourly:
        if t in trusted:
            continue
        if looks_numeric_tag(str(t)):
            continue
        trusted.add(str(t))

    all_tags = sorted(trusted)
    return all_tags


# ----------------- styling helpers -----------------

CARD_STYLE = {
    "backgroundColor": "#1c2026",
    "borderRadius": "10px",
    "border": "1px solid #2b323c",
    "padding": "12px 16px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "8px",
}

GAUGE_ROW_STYLE = {
    "display": "flex",
    "gap": "12px",
    "flexWrap": "wrap",   # allow wrapping on very narrow screens
}

GAUGE_CONTAINER_STYLE = {
    "flex": "1",
    "textAlign": "center",
}


def build_tag_card(
    tag: str,
    latest: Dict[str, Tuple[float, str]],
    last_hour_stats: Dict[str, Tuple[float, datetime, datetime, int]],
    current_hour_avg: Dict[str, float],
    thresholds: Dict[str, Dict[str, float]],
) -> html.Div:
    """
    Build a card with three gauges for a single tag:
        - Current Value (falls back to last full hour if no live sample)
        - Last Full Hour
        - Live Hourly Average (current hour so far)
    """
    # base values
    cur_val, cur_plc_status = latest.get(tag, (float("nan"), "No data"))
    last_avg, hs, he, sample_count = last_hour_stats.get(
        tag, (float("nan"), None, None, 0)
    )
    live_hr_val = current_hour_avg.get(tag, float("nan"))

    # If we have no live "current" value but we do have a last full hour,
    # use that for the current gauge so it doesn't display "non-numeric".
    derived_current = False
    if cur_val != cur_val and last_avg == last_avg:  # NaN current, finite last_avg
        cur_val = last_avg
        cur_plc_status = "Derived from last full hour"
        derived_current = True

    # Thresholds (if any)
    low = thresholds.get(tag, {}).get("low")
    high = thresholds.get(tag, {}).get("high")

    # Derive gauge ranges
    sample_vals = [cur_val, last_avg, live_hr_val]
    low_eff, high_eff, gmin, gmax = compute_gauge_range(low, high, sample_vals)

    # Classification
    cur_status_eval = classify_value(cur_val, low_eff, high_eff)
    last_status_eval = classify_value(last_avg, low_eff, high_eff)
    live_hr_status_eval = classify_value(live_hr_val, low_eff, high_eff)

    cur_color = status_color(cur_status_eval)
    last_color = status_color(last_status_eval)
    live_hr_color = status_color(live_hr_status_eval)

    # Labels with 2 decimal places
    if cur_val == cur_val:
        prefix = "Current"
        if derived_current:
            prefix = "Current (from last full hour)"
        cur_label_text = (
            f"{prefix}: {cur_val:.2f} (PLC: {cur_plc_status}, eval: {cur_status_eval})"
        )
    else:
        cur_label_text = f"Current: no data (PLC: {cur_plc_status})"

    if last_avg == last_avg and hs is not None and he is not None and sample_count > 0:
        try:
            hs_s = hs.strftime("%Y-%m-%d %H:%M")
            he_s = he.strftime("%H:%M")
        except Exception:
            hs_s = str(hs)
            he_s = str(he)
        last_label_text = (
            f"Last full hour {hs_s}-{he_s}: {last_avg:.2f} "
            f"(samples: {sample_count}, eval: {last_status_eval})"
        )
    else:
        last_label_text = "Last full hour: no data"

    if live_hr_val == live_hr_val:
        live_hr_label_text = (
            f"Live hourly average: {live_hr_val:.2f} "
            f"(current hour, eval: {live_hr_status_eval})"
        )
    else:
        live_hr_label_text = "Live hourly average: no data"

    gauge_size = 170

    current_gauge = html.Div(
        style=GAUGE_CONTAINER_STYLE,
        children=[
            html.Div("Current", style={"marginBottom": "4px", "fontWeight": "600"}),
            daq.Gauge(
                id={"type": "gauge", "tag": tag, "kind": "current"},
                min=gmin,
                max=gmax,
                value=cur_val if cur_val == cur_val else gmin,
                showCurrentValue=True,
                color=cur_color,
                label="",
                size=gauge_size,
                units="",
            ),
            html.Div(cur_label_text, style={"fontSize": "11px", "marginTop": "4px"}),
        ],
    )

    last_hour_gauge = html.Div(
        style=GAUGE_CONTAINER_STYLE,
        children=[
            html.Div("Last Full Hour", style={"marginBottom": "4px", "fontWeight": "600"}),
            daq.Gauge(
                id={"type": "gauge", "tag": tag, "kind": "last_hour"},
                min=gmin,
                max=gmax,
                value=last_avg if last_avg == last_avg else gmin,
                showCurrentValue=True,
                color=last_color,
                label="",
                size=gauge_size,
                units="",
            ),
            html.Div(last_label_text, style={"fontSize": "11px", "marginTop": "4px"}),
        ],
    )

    live_hour_gauge = html.Div(
        style=GAUGE_CONTAINER_STYLE,
        children=[
            html.Div(
                "Live Hourly Average",
                style={"marginBottom": "4px", "fontWeight": "600"},
            ),
            daq.Gauge(
                id={"type": "gauge", "tag": tag, "kind": "current_hour"},
                min=gmin,
                max=gmax,
                value=live_hr_val if live_hr_val == live_hr_val else gmin,
                showCurrentValue=True,
                color=live_hr_color,
                label="",
                size=gauge_size,
                units="",
            ),
            html.Div(live_hr_label_text, style={"fontSize": "11px", "marginTop": "4px"}),
        ],
    )

    header = html.Div(
        [
            html.Div(
                tag,
                style={
                    "fontWeight": "600",
                    "fontSize": "13px",
                    "color": "#e0e6ed",
                },
            ),
            html.Div(
                f"Thresholds: [{low if low is not None else 'auto'}, "
                f"{high if high is not None else 'auto'}]",
                style={"fontSize": "11px", "color": "#90a4ae"},
            ),
        ]
    )

    gauges_row = html.Div(
        style=GAUGE_ROW_STYLE,
        children=[current_gauge, last_hour_gauge, live_hour_gauge],
    )

    return html.Div(style=CARD_STYLE, children=[header, gauges_row])


# ----------------- dash app -----------------

app = Dash(__name__)
app.title = "CIP Web Dashboard"
server = app.server

app.layout = html.Div(
    style={
        "backgroundColor": "#16191f",
        "color": "#ecf0f1",
        "fontFamily": "Segoe UI, sans-serif",
        "padding": "16px",
        "minHeight": "100vh",
    },
    children=[
        # Header
        html.Div(
            style={
                "backgroundColor": "#20242b",
                "borderRadius": "8px",
                "padding": "12px 16px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "12px",
            },
            children=[
                html.Div(
                    children=[
                        html.Div(
                            "CIP Tag Web Dashboard",
                            style={
                                "fontSize": "18px",
                                "fontWeight": "700",
                            },
                        ),
                        html.Div(
                            "Live view of PLC tags and hourly aggregates",
                            style={
                                "fontSize": "11px",
                                "color": "#90a4ae",
                            },
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            id="config-version-label",
                            style={
                                "fontSize": "11px",
                                "color": "#b0bec5",
                                "textAlign": "right",
                            },
                        ),
                        html.Div(
                            id="last-update-label",
                            style={
                                "fontSize": "11px",
                                "color": "#b0bec5",
                                "textAlign": "right",
                            },
                        ),
                    ]
                ),
            ],
        ),

        dcc.Interval(
            id="refresh-interval",
            interval=REFRESH_MS,
            n_intervals=0,
        ),

        # Tabs: Overview (gauges) & Thresholds (editor)
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            colors={
                "border": "#2b323c",
                "primary": "#3498db",
                "background": "#1c2026",
            },
            style={
                "borderRadius": "8px",
                "overflow": "hidden",
            },
            children=[
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    style={"backgroundColor": "#1c2026", "color": "#b0bec5"},
                    selected_style={
                        "backgroundColor": "#20242b",
                        "color": "white",
                        "fontWeight": "600",
                    },
                    children=[
                        html.Div(
                            id="tag-cards-container",
                            style={
                                "padding": "12px",
                                # one tag card per row (vertical stack)
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "14px",
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Thresholds",
                    value="thresholds",
                    style={"backgroundColor": "#1c2026", "color": "#b0bec5"},
                    selected_style={
                        "backgroundColor": "#20242b",
                        "color": "white",
                        "fontWeight": "600",
                    },
                    children=[
                        html.Div(
                            style={
                                "padding": "12px 16px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "8px",
                                "maxWidth": "380px",
                            },
                            children=[
                                html.Div(
                                    "Threshold Editor",
                                    style={
                                        "fontSize": "14px",
                                        "fontWeight": "600",
                                    },
                                ),
                                html.Div(
                                    "Select a tag and define its good operating range. "
                                    "Gauges on the Overview tab will color based on these thresholds.",
                                    style={"fontSize": "11px", "color": "#90a4ae"},
                                ),
                                html.Label(
                                    "Tag:",
                                    style={"marginTop": "6px", "fontSize": "11px"},
                                ),
                                dcc.Dropdown(
                                    id="threshold-tag-dropdown",
                                    options=[],
                                    value=None,
                                    placeholder="No tags yet",
                                    style={"color": "#000"},
                                ),
                                html.Label(
                                    "Good Range (low / high):",
                                    style={"marginTop": "6px", "fontSize": "11px"},
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "6px"},
                                    children=[
                                        dcc.Input(
                                            id="threshold-low-input",
                                            type="number",
                                            placeholder="Low",
                                            style={"flex": 1},
                                        ),
                                        dcc.Input(
                                            id="threshold-high-input",
                                            type="number",
                                            placeholder="High",
                                            style={"flex": 1},
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "Save Thresholds",
                                    id="threshold-save-btn",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "6px",
                                        "backgroundColor": "#3498db",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "6px 10px",
                                        "borderRadius": "6px",
                                        "fontSize": "11px",
                                    },
                                ),
                                html.Div(
                                    id="threshold-save-status",
                                    style={"fontSize": "11px", "color": "#b0bec5"},
                                ),
                                html.Div(
                                    f"File: {THRESHOLDS_JSON}",
                                    style={
                                        "marginTop": "8px",
                                        "fontSize": "9px",
                                        "color": "#78909c",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ----------------- callbacks -----------------


@app.callback(
    Output("threshold-tag-dropdown", "options"),
    Output("threshold-tag-dropdown", "value"),
    Input("refresh-interval", "n_intervals"),
    State("threshold-tag-dropdown", "value"),
)
def refresh_threshold_dropdown(n, current_value):
    """
    Refresh list of tags available for threshold editing.
    Keeps current selection if still valid.
    """
    try:
        tags = discover_all_tags_for_dropdown()
    except Exception:
        tags = []

    options = [{"label": t, "value": t} for t in tags]
    if current_value in tags:
        value = current_value
    else:
        value = tags[0] if tags else None
    return options, value


@app.callback(
    Output("threshold-low-input", "value"),
    Output("threshold-high-input", "value"),
    Input("threshold-tag-dropdown", "value"),
)
def populate_threshold_inputs(tag):
    """Fill threshold inputs when tag changes."""
    th = load_thresholds()
    if not tag or tag not in th:
        return None, None
    return th[tag].get("low"), th[tag].get("high")


@app.callback(
    Output("threshold-save-status", "children"),
    Input("threshold-save-btn", "n_clicks"),
    State("threshold-tag-dropdown", "value"),
    State("threshold-low-input", "value"),
    State("threshold-high-input", "value"),
    prevent_initial_call=True,
)
def save_thresholds_callback(n_clicks, tag, low, high):
    if not tag:
        return "Select a tag first."
    if low is None or high is None:
        return "Enter both low and high values."
    try:
        low = float(low)
        high = float(high)
    except ValueError:
        return "Low / High must be numeric."

    if high < low:
        low, high = high, low

    th = load_thresholds()
    old_entry = th.get(tag, {})
    old_low = old_entry.get("low") if isinstance(old_entry, dict) else None
    old_high = old_entry.get("high") if isinstance(old_entry, dict) else None
    th[tag] = {"low": low, "high": high}
    save_thresholds(th)
    if old_low != low or old_high != high:
        log_config_change(
            field=f"Thresholds: {tag}",
            old_value=old_entry,
            new_value=th[tag],
            reason="Edited in web dashboard",
        )
    return f"Saved thresholds for {tag}: [{low}, {high}]"


@app.callback(
    Output("tag-cards-container", "children"),
    Output("last-update-label", "children"),
    Output("config-version-label", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_dashboard(n):
    """
    Main refresh callback.
    Fully defensive: on error, returns an error card rather than crashing.
    """
    try:
        raw_df = load_latest_raw_df()
        hourly_df = load_hourly_stats()
        thresholds = load_thresholds()
        config_version = compute_config_version_text()

        latest_by_tag, current_hour_avg, last_ts = extract_raw_stats(raw_df)
        last_hour_stats = extract_last_full_hour(hourly_df)

        tags_raw = set(latest_by_tag.keys())
        tags_hourly = set(last_hour_stats.keys())
        tags_cfg = set(load_configured_tags_from_settings())
        all_tags = sorted((tags_raw | tags_hourly | tags_cfg), key=str)

        # Filter out numeric-looking tags that somehow slipped through
        all_tags = [t for t in all_tags if not looks_numeric_tag(str(t))]

        if not all_tags:
            msg = "No tags found yet. Waiting for data from CIP Tag Poller..."
            empty_card = html.Div(
                style=CARD_STYLE,
                children=[
                    html.Div(
                        "No Tag Data",
                        style={
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        msg,
                        style={
                            "fontSize": "11px",
                            "color": "#90a4ae",
                        },
                    ),
                ],
            )
            last_update = "Last update: no data"
            return [empty_card], last_update, config_version

        cards = []
        for tag in all_tags:
            card = build_tag_card(
                tag=tag,
                latest=latest_by_tag,
                last_hour_stats=last_hour_stats,
                current_hour_avg=current_hour_avg,
                thresholds=thresholds,
            )
            cards.append(card)

        if last_ts is not None:
            try:
                last_update_str = last_ts.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                last_update_str = str(last_ts)
            last_update = f"Last update: {last_update_str}"
        else:
            last_update = "Last update: unknown"

        return cards, last_update, config_version

    except Exception as e:
        error_card = html.Div(
            style=CARD_STYLE,
            children=[
                html.Div(
                    "Dashboard Error",
                    style={
                        "fontWeight": "600",
                        "fontSize": "13px",
                        "marginBottom": "4px",
                        "color": "#ef9a9a",
                    },
                ),
                html.Div(
                    f"An error occurred while updating the dashboard: {e}",
                    style={
                        "fontSize": "11px",
                        "color": "#ef9a9a",
                    },
                ),
            ],
        )
        last_update = "Last update: error"
        return [error_card], last_update, compute_config_version_text()


# ----------------- main -----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
