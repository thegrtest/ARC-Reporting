#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dash web dashboard for CIP Tag Poller

Reads:
    logs/raw_data.csv
    logs/hourly_averages.csv
and exposes a network dashboard with live gauges + hourly stats.

Run:
    python cip_dash_server.py

Then open:
    http://<this_machine_ip>:8050  (on your network)
"""

import os
import json
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_daq as daq


# ----------------- config -----------------

LOG_DIR = "logs"
RAW_CSV = os.path.join(LOG_DIR, "raw_data.csv")
HOURLY_CSV = os.path.join(LOG_DIR, "hourly_averages.csv")
THRESHOLDS_JSON = os.path.join(LOG_DIR, "thresholds.json")

REFRESH_MS = 5000  # dashboard refresh interval

os.makedirs(LOG_DIR, exist_ok=True)


# ----------------- helpers -----------------

def load_thresholds() -> Dict[str, Dict[str, float]]:
    """Load per-tag thresholds from JSON, or return empty dict."""
    if not os.path.exists(THRESHOLDS_JSON):
        return {}
    try:
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        # expected shape: {tag: {"low": float, "high": float}}
        return data
    except Exception:
        return {}


def save_thresholds(thresholds: Dict[str, Dict[str, float]]):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)


def load_latest_values() -> Dict[str, Tuple[float, str]]:
    """
    Return {tag: (value, status)} for latest row per tag from raw_data.csv.
    Non-numeric values are skipped for value but status is still shown.
    """
    if not os.path.exists(RAW_CSV):
        return {}

    try:
        df = pd.read_csv(RAW_CSV)
    except Exception:
        return {}

    if df.empty:
        return {}

    # Keep only required columns
    cols = {"timestamp", "tag", "value", "status"}
    df = df[[c for c in df.columns if c in cols]]

    # Sort by timestamp and take last row per tag
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["tag", "timestamp"])
    except Exception:
        # fallback: just use index order
        df = df.sort_index()

    latest = {}
    for tag, group in df.groupby("tag"):
        row = group.iloc[-1]
        status = str(row.get("status", ""))
        try:
            val = float(row["value"])
        except Exception:
            val = float("nan")
        latest[tag] = (val, status)

    return latest


def load_hourly_stats():
    """
    Load hourly averages as DataFrame with columns:
        hour_start, hour_end, tag, avg_value, sample_count
    """
    if not os.path.exists(HOURLY_CSV):
        return pd.DataFrame(
            columns=["hour_start", "hour_end", "tag", "avg_value", "sample_count"]
        )
    try:
        df = pd.read_csv(HOURLY_CSV)
        if "hour_start" in df.columns:
            df["hour_start"] = pd.to_datetime(df["hour_start"], errors="coerce")
        if "hour_end" in df.columns:
            df["hour_end"] = pd.to_datetime(df["hour_end"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(
            columns=["hour_start", "hour_end", "tag", "avg_value", "sample_count"]
        )


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

    # within 10% beyond low/high: warning
    width = max(1e-6, high - low)
    if value < low and (low - value) <= 0.1 * width:
        return "warning"
    if value > high and (value - high) <= 0.1 * width:
        return "warning"

    return "bad"


def status_color(status: str) -> str:
    if status == "good":
        return "#2ecc71"  # green
    if status == "warning":
        return "#f1c40f"  # yellow
    if status == "bad":
        return "#e74c3c"  # red
    return "#7f8c8d"      # gray


# ----------------- dash app -----------------

app = Dash(__name__)
app.title = "CIP Web Dashboard"

server = app.server  # so you can host with gunicorn if needed

# initial data
latest_values = load_latest_values()
thresholds = load_thresholds()
tags_sorted = sorted(latest_values.keys())

default_tag = tags_sorted[0] if tags_sorted else ""

app.layout = html.Div(
    style={
        "backgroundColor": "#1e1e1e",
        "color": "#ecf0f1",
        "fontFamily": "Segoe UI, sans-serif",
        "padding": "16px",
    },
    children=[
        html.H2("CIP Tag Web Dashboard"),
        html.Div(
            style={"display": "flex", "gap": "24px"},
            children=[
                # Left column: controls
                html.Div(
                    style={"flex": "0 0 320px"},
                    children=[
                        html.H4("Tag Selection & Thresholds"),
                        dcc.Dropdown(
                            id="tag-dropdown",
                            options=[
                                {"label": t, "value": t} for t in tags_sorted
                            ],
                            value=default_tag,
                            placeholder="Select a tag",
                            style={"color": "#000"},
                        ),
                        html.Br(),
                        html.Div(
                            [
                                html.Label("Good Range (low / high):"),
                                html.Div(
                                    style={"display": "flex", "gap": "8px"},
                                    children=[
                                        dcc.Input(
                                            id="low-input",
                                            type="number",
                                            placeholder="Low",
                                            style={"flex": 1},
                                        ),
                                        dcc.Input(
                                            id="high-input",
                                            type="number",
                                            placeholder="High",
                                            style={"flex": 1},
                                        ),
                                    ],
                                ),
                                html.Br(),
                                html.Button(
                                    "Save Thresholds for Tag",
                                    id="save-threshold-btn",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#3498db",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "6px 12px",
                                    },
                                ),
                                html.Div(
                                    id="save-status",
                                    style={"marginTop": "8px", "fontSize": "12px"},
                                ),
                            ]
                        ),
                        html.Hr(style={"borderColor": "#555"}),
                        html.Div(
                            [
                                html.Label("Last update:"),
                                html.Div(id="last-update-label"),
                            ]
                        ),
                        dcc.Interval(
                            id="refresh-interval",
                            interval=REFRESH_MS,
                            n_intervals=0,
                        ),
                    ],
                ),
                # Right column: gauges + table
                html.Div(
                    style={"flex": 1},
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "24px"},
                            children=[
                                html.Div(
                                    style={"flex": 1},
                                    children=[
                                        html.H4("Current Value"),
                                        daq.Gauge(
                                            id="current-gauge",
                                            min=0,
                                            max=100,
                                            value=0,
                                            showCurrentValue=True,
                                            color="#7f8c8d",
                                            label="",
                                            size=230,
                                            units="",
                                        ),
                                        html.Div(
                                            id="current-status-label",
                                            style={"textAlign": "center", "marginTop": "8px"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"flex": 1},
                                    children=[
                                        html.H4("Last Hour Average"),
                                        daq.Gauge(
                                            id="last-hour-gauge",
                                            min=0,
                                            max=100,
                                            value=0,
                                            showCurrentValue=True,
                                            color="#7f8c8d",
                                            label="",
                                            size=230,
                                            units="",
                                        ),
                                        html.Div(
                                            id="last-hour-label",
                                            style={"textAlign": "center", "marginTop": "8px"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"flex": 1},
                                    children=[
                                        html.H4("Current Hour Average (So Far)"),
                                        daq.Gauge(
                                            id="current-hour-gauge",
                                            min=0,
                                            max=100,
                                            value=0,
                                            showCurrentValue=True,
                                            color="#7f8c8d",
                                            label="",
                                            size=230,
                                            units="",
                                        ),
                                        html.Div(
                                            id="current-hour-label",
                                            style={"textAlign": "center", "marginTop": "8px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Hr(style={"borderColor": "#555", "marginTop": "24px"}),
                        html.H4("Hourly Averages (per tag)"),
                        dcc.Loading(
                            type="dot",
                            color="#3498db",
                            children=html.Div(id="hourly-table"),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ----------------- callbacks -----------------


@app.callback(
    Output("low-input", "value"),
    Output("high-input", "value"),
    Input("tag-dropdown", "value"),
)
def populate_threshold_inputs(tag):
    """Fill threshold inputs when tag changes."""
    th = load_thresholds()
    if not tag or tag not in th:
        return None, None
    return th[tag].get("low"), th[tag].get("high")


@app.callback(
    Output("save-status", "children"),
    Input("save-threshold-btn", "n_clicks"),
    State("tag-dropdown", "value"),
    State("low-input", "value"),
    State("high-input", "value"),
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
    th = load_thresholds()
    th[tag] = {"low": low, "high": high}
    save_thresholds(th)
    return f"Saved thresholds for {tag}: [{low}, {high}]"


@app.callback(
    Output("current-gauge", "value"),
    Output("current-gauge", "min"),
    Output("current-gauge", "max"),
    Output("current-gauge", "color"),
    Output("current-status-label", "children"),
    Output("last-hour-gauge", "value"),
    Output("last-hour-gauge", "min"),
    Output("last-hour-gauge", "max"),
    Output("last-hour-gauge", "color"),
    Output("last-hour-label", "children"),
    Output("current-hour-gauge", "value"),
    Output("current-hour-gauge", "min"),
    Output("current-hour-gauge", "max"),
    Output("current-hour-gauge", "color"),
    Output("current-hour-label", "children"),
    Output("hourly-table", "children"),
    Output("last-update-label", "children"),
    Input("refresh-interval", "n_intervals"),
    State("tag-dropdown", "value"),
)
def update_live_data(n, tag):
    latest = load_latest_values()
    hourly_df = load_hourly_stats()
    thresholds = load_thresholds()

    # default gauge values
    cur_val = 0.0
    cur_min = 0.0
    cur_max = 1.0
    cur_color = "#7f8c8d"
    cur_label = "No data"

    last_val = 0.0
    last_min = 0.0
    last_max = 1.0
    last_color = "#7f8c8d"
    last_label = "No data"

    curr_hr_val = 0.0
    curr_hr_min = 0.0
    curr_hr_max = 1.0
    curr_hr_color = "#7f8c8d"
    curr_hr_label = "No data"

    last_update = "No data"

    # Table placeholder
    table_children = html.Div("No hourly averages yet.")

    if not tag:
        return (
            cur_val, cur_min, cur_max, cur_color, cur_label,
            last_val, last_min, last_max, last_color, last_label,
            curr_hr_val, curr_hr_min, curr_hr_max, curr_hr_color, curr_hr_label,
            table_children,
            last_update,
        )

    # thresholds
    low = thresholds.get(tag, {}).get("low", 0.0)
    high = thresholds.get(tag, {}).get("high", 100.0)

    # 1) current value
    if tag in latest:
        v, status = latest[tag]
        if v == v:  # not NaN
            cur_val = v
            span = max(1.0, abs(high - low) * 1.5)
            center = (high + low) / 2.0
            cur_min = center - span / 2.0
            cur_max = center + span / 2.0
            val_status = classify_value(v, low, high)
            cur_color = status_color(val_status)
            cur_label = f"Current: {v:.4f}  (PLC status: {status}, eval: {val_status})"
        else:
            cur_label = f"Current: non-numeric value (PLC status: {status})"
        # last update time
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2) last hour average & current hour average (from hourly_df)
    if not hourly_df.empty:
        tag_df = hourly_df[hourly_df["tag"] == tag].copy()
        if not tag_df.empty:
            # last full hour: row with max hour_start
            tag_df = tag_df.sort_values("hour_start")
            last_row = tag_df.iloc[-1]
            lv = float(last_row["avg_value"])
            last_val = lv
            span = max(1.0, abs(high - low) * 1.5)
            center = (high + low) / 2.0
            last_min = center - span / 2.0
            last_max = center + span / 2.0
            val_status = classify_value(lv, low, high)
            last_color = status_color(val_status)
            hs = last_row["hour_start"]
            he = last_row["hour_end"]
            try:
                hs_str = hs.strftime("%H:%M")
                he_str = he.strftime("%H:%M")
            except Exception:
                hs_str = str(hs)
                he_str = str(he)
            last_label = f"{hs_str}-{he_str} avg: {lv:.4f} (eval: {val_status})"

            # Table for all hours
            rows = []
            for _, r in tag_df.iterrows():
                try:
                    hs_s = r["hour_start"].strftime("%Y-%m-%d %H:%M")
                    he_s = r["hour_end"].strftime("%H:%M")
                except Exception:
                    hs_s = str(r["hour_start"])
                    he_s = str(r["hour_end"])
                rows.append(
                    html.Tr(
                        [
                            html.Td(hs_s),
                            html.Td(he_s),
                            html.Td(f"{float(r['avg_value']):.4f}"),
                            html.Td(int(r.get("sample_count", 0))),
                        ]
                    )
                )
            table_children = html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Hour Start"),
                                html.Th("Hour End"),
                                html.Th("Avg Value"),
                                html.Th("Samples"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                style={
                    "width": "100%",
                    "borderCollapse": "collapse",
                },
            )

    # 3) current hour average (so far)
    # We approximate by using the last hour in hourly_df if it is the current hour,
    # but your PySide app writes only completed hours. If you want "current hour
    # so far", you can later add another CSV or JSON; for now we just reuse last hour.
    curr_hr_val = last_val
    curr_hr_min = last_min
    curr_hr_max = last_max
    curr_hr_color = last_color
    curr_hr_label = "Using last full hour as approx (see note in code)."

    return (
        cur_val, cur_min, cur_max, cur_color, cur_label,
        last_val, last_min, last_max, last_color, last_label,
        curr_hr_val, curr_hr_min, curr_hr_max, curr_hr_color, curr_hr_label,
        table_children,
        last_update,
    )


# ----------------- main -----------------

if __name__ == "__main__":
    # host="0.0.0.0" lets you open from other machines on the LAN
    app.run_server(host="0.0.0.0", port=8050, debug=False)
