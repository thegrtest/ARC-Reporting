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
ENV_EVENTS_CSV = os.path.join(LOG_DIR, "env_events.csv")
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
    base_cols = [
        "timestamp",
        "date",
        "time",
        "tag",
        "alias",
        "value",
        "status",
        "qa_flag",
    ]

    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=base_cols)

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=base_cols)

    # Normalize columns
    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception:
        df["timestamp"] = pd.NaT

    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    df["status_str"] = df["status"].astype(str)
    df["qa_flag"] = df.get("qa_flag", "OK").fillna("OK")
    df["ok"] = df["qa_flag"].astype(str).str.upper().eq("OK")

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


def load_env_events() -> pd.DataFrame:
    """Load environmental / data quality events (including DATA_GAP)."""
    if not os.path.exists(ENV_EVENTS_CSV):
        return pd.DataFrame(columns=["timestamp", "event_type", "tag", "duration_sec"])
    try:
        df = pd.read_csv(ENV_EVENTS_CSV)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "event_type", "tag", "duration_sec"])

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            df["timestamp"] = pd.NaT
    else:
        df["timestamp"] = pd.NaT

    if "duration_sec" in df.columns:
        try:
            df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
        except Exception:
            df["duration_sec"] = None

    return df


def _to_float(val: object) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_threshold_entry(raw_entry: object) -> Dict[str, object]:
    """Normalize a raw threshold entry into the new schema."""
    if not isinstance(raw_entry, dict):
        return {}

    entry: Dict[str, object] = {}

    # carry friendly metadata if present
    for key in ["alias", "units"]:
        if isinstance(raw_entry.get(key), str) and raw_entry.get(key).strip():
            entry[key] = raw_entry.get(key).strip()

    # prefer new keys, fall back to legacy low/high
    low_oper = _to_float(raw_entry.get("low_oper"))
    if low_oper is None:
        low_oper = _to_float(raw_entry.get("low"))

    high_oper = _to_float(raw_entry.get("high_oper"))
    if high_oper is None:
        high_oper = _to_float(raw_entry.get("high"))

    low_limit = _to_float(raw_entry.get("low_limit"))
    high_limit = _to_float(raw_entry.get("high_limit"))

    if low_oper is not None:
        entry["low_oper"] = low_oper
    if high_oper is not None:
        entry["high_oper"] = high_oper
    if low_limit is not None:
        entry["low_limit"] = low_limit
    if high_limit is not None:
        entry["high_limit"] = high_limit

    return entry


def load_thresholds() -> Dict[str, Dict[str, object]]:
    """
    Load per-tag thresholds/limits from JSON if available.
    thresholds.json format:
        {
          "TagName": {
              "alias": "Kiln Temperature",
              "units": "°C",
              "low_oper": 800,
              "high_oper": 950,
              "low_limit": 750,
              "high_limit": 1000
          },
          ...
        }

    Legacy `{ "Tag": {"low": x, "high": y} }` entries are normalized to
    `low_oper`/`high_oper`.
    """
    if not os.path.exists(THRESHOLDS_JSON):
        return {}
    try:
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            normalized = {k: _normalize_threshold_entry(v) for k, v in data.items()}
            return normalized
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

    if "qa_flag" not in raw_df.columns:
        raw_df["qa_flag"] = "OK"
    raw_df["ok"] = raw_df["qa_flag"].astype(str).str.upper().eq("OK")

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


def compute_quality_metrics(
    raw_df: pd.DataFrame, events_df: pd.DataFrame, tags: List[str]
) -> Dict[str, Dict[str, float]]:
    """Return per-tag metrics: {tag: {valid_pct, gap_count, longest_gap}}"""
    stats: Dict[str, Dict[str, float]] = {
        str(tag): {"valid_pct": 0.0, "gap_count": 0, "longest_gap": 0.0}
        for tag in tags
    }

    cutoff = datetime.now() - timedelta(hours=24)

    if not raw_df.empty and "timestamp" in raw_df.columns:
        try:
            recent = raw_df[raw_df["timestamp"] >= cutoff]
        except Exception:
            recent = raw_df

        if not recent.empty:
            valid_mask = recent.get("qa_flag", "OK").astype(str).str.upper().eq("OK")
            total_counts = recent.groupby("tag").size()
            valid_counts = recent[valid_mask].groupby("tag").size()
            for tag, total in total_counts.items():
                tag_str = str(tag)
                ok_count = float(valid_counts.get(tag, 0))
                if total > 0:
                    stats.setdefault(tag_str, {"valid_pct": 0.0, "gap_count": 0, "longest_gap": 0.0})[
                        "valid_pct"
                    ] = round((ok_count / float(total)) * 100.0, 1)

    if not events_df.empty and "timestamp" in events_df.columns:
        try:
            recent_events = events_df[events_df["timestamp"] >= cutoff]
        except Exception:
            recent_events = events_df

        gap_events = recent_events[
            recent_events["event_type"].astype(str).str.upper() == "DATA_GAP"
        ]
        if not gap_events.empty:
            counts = gap_events.groupby("tag").size()
            longest = gap_events.groupby("tag")["duration_sec"].max()
            for tag, cnt in counts.items():
                tag_str = str(tag)
                stat = stats.setdefault(tag_str, {"valid_pct": 0.0, "gap_count": 0, "longest_gap": 0.0})
                stat["gap_count"] = int(cnt)
                try:
                    stat["longest_gap"] = float(longest.get(tag, 0.0) or 0.0)
                except Exception:
                    stat["longest_gap"] = 0.0

    return stats


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
    thresholds: Dict[str, Dict[str, object]],
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

    entry = thresholds.get(tag, {}) if isinstance(thresholds.get(tag, {}), dict) else {}

    alias = entry.get("alias") if isinstance(entry.get("alias"), str) else None
    units = entry.get("units") if isinstance(entry.get("units"), str) else None

    low_oper = entry.get("low_oper")
    high_oper = entry.get("high_oper")
    low_limit = entry.get("low_limit")
    high_limit = entry.get("high_limit")

    # Derive gauge ranges using operational thresholds first, then limits, then data
    sample_vals = [cur_val, last_avg, live_hr_val]
    low_for_range = low_oper if low_oper is not None else low_limit
    high_for_range = high_oper if high_oper is not None else high_limit
    low_eff, high_eff, gmin, gmax = compute_gauge_range(
        low_for_range, high_for_range, sample_vals
    )

    # When classifying, prefer operational thresholds if set; otherwise use limits/range
    low_for_class = low_oper if low_oper is not None else low_eff
    high_for_class = high_oper if high_oper is not None else high_eff

    # Classification
    cur_status_eval = classify_value(cur_val, low_for_class, high_for_class)
    last_status_eval = classify_value(last_avg, low_for_class, high_for_class)
    live_hr_status_eval = classify_value(live_hr_val, low_for_class, high_for_class)

    cur_color = status_color(cur_status_eval)
    last_color = status_color(last_status_eval)
    live_hr_color = status_color(live_hr_status_eval)

    # Hard-limit exceedance text (if provided)
    def limit_note(value: float) -> str:
        if value != value:
            return ""
        if low_limit is not None and value < low_limit:
            return f" (below limit {low_limit})"
        if high_limit is not None and value > high_limit:
            return f" (above limit {high_limit})"
        return ""

    # Labels with 2 decimal places
    if cur_val == cur_val:
        prefix = "Current"
        if derived_current:
            prefix = "Current (from last full hour)"
        cur_label_text = (
            f"{prefix}: {cur_val:.2f} (PLC: {cur_plc_status}, eval: {cur_status_eval})"
            f"{limit_note(cur_val)}"
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
            f"{limit_note(last_avg)}"
        )
    else:
        last_label_text = "Last full hour: no data"

    if live_hr_val == live_hr_val:
        live_hr_label_text = (
            f"Live hourly average: {live_hr_val:.2f} "
            f"(current hour, eval: {live_hr_status_eval})"
            f"{limit_note(live_hr_val)}"
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

    display_name = alias or tag
    subtitle_bits = []
    if alias:
        subtitle_bits.append(f"Tag: {tag}")
    if units:
        subtitle_bits.append(f"Units: {units}")
    subtitle = " • ".join(subtitle_bits)

    def format_bounds(label: str, low_val: Optional[float], high_val: Optional[float], fallback: str) -> str:
        bounds: List[str] = []
        if low_val is not None:
            bounds.append(str(low_val))
        if high_val is not None:
            bounds.append(str(high_val))
        if bounds:
            return f"{label}: [{' – '.join(bounds)}]"
        return fallback

    oper_text = format_bounds("Operational", low_oper, high_oper, "Operational: auto")
    limit_text = format_bounds(
        "Regulatory limit", low_limit, high_limit, "Regulatory limit: none set"
    )

    header_children = [
        html.Div(
            display_name,
            style={
                "fontWeight": "600",
                "fontSize": "13px",
                "color": "#e0e6ed",
            },
        ),
        html.Div(
            [
                html.Div(oper_text),
                html.Div(limit_text),
                html.Div(subtitle) if subtitle else None,
            ],
            style={"fontSize": "11px", "color": "#90a4ae", "lineHeight": "16px"},
        ),
    ]

    header = html.Div([child for child in header_children if child is not None])

    gauges_row = html.Div(
        style=GAUGE_ROW_STYLE,
        children=[current_gauge, last_hour_gauge, live_hour_gauge],
    )

    return html.Div(style=CARD_STYLE, children=[header, gauges_row])


def build_quality_card(stats: Dict[str, Dict[str, float]]) -> html.Div:
    rows = []
    for tag in sorted(stats.keys(), key=str):
        tag_stats = stats.get(tag, {})
        valid_pct = tag_stats.get("valid_pct", 0.0)
        gap_count = int(tag_stats.get("gap_count", 0) or 0)
        longest_gap = tag_stats.get("longest_gap", 0.0) or 0.0
        rows.append(
            html.Tr(
                children=[
                    html.Td(tag, style={"fontWeight": "600", "fontSize": "11px"}),
                    html.Td(f"{valid_pct:.1f}%", style={"fontSize": "11px"}),
                    html.Td(str(gap_count), style={"fontSize": "11px"}),
                    html.Td(f"{longest_gap:.0f}s", style={"fontSize": "11px"}),
                ]
            )
        )

    table = html.Table(
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "11px",
        },
        children=[
            html.Thead(
                html.Tr(
                    children=[
                        html.Th("Tag", style={"textAlign": "left", "paddingBottom": "4px"}),
                        html.Th(
                            "% Valid (last 24h)",
                            style={"textAlign": "left", "paddingBottom": "4px"},
                        ),
                        html.Th(
                            "Gap count", style={"textAlign": "left", "paddingBottom": "4px"}
                        ),
                        html.Th(
                            "Longest gap", style={"textAlign": "left", "paddingBottom": "4px"}
                        ),
                    ]
                )
            ),
            html.Tbody(rows if rows else [html.Tr(html.Td("No data yet"))]),
        ],
    )

    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(
                "Data Quality (last 24h)",
                style={"fontWeight": "600", "fontSize": "13px"},
            ),
            html.Div(
                "QA flags drive valid %; DATA_GAP events summarize missing runs.",
                style={"fontSize": "11px", "color": "#90a4ae"},
            ),
            table,
        ],
    )


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
                                    "Threshold / Limit Editor",
                                    style={
                                        "fontSize": "14px",
                                        "fontWeight": "600",
                                    },
                                ),
                                html.Div(
                                    "Separate operator warning bands from regulatory/permit limits. "
                                    "Operational thresholds drive gauge coloring; limits track strict compliance.",
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
                                    "Alias (friendly name):",
                                    style={"marginTop": "6px", "fontSize": "11px"},
                                ),
                                dcc.Input(
                                    id="threshold-alias-input",
                                    type="text",
                                    placeholder="Kiln Temperature",
                                    style={"width": "100%"},
                                ),
                                html.Label(
                                    "Units:",
                                    style={"marginTop": "6px", "fontSize": "11px"},
                                ),
                                dcc.Input(
                                    id="threshold-units-input",
                                    type="text",
                                    placeholder="°C",
                                    style={"width": "100%"},
                                ),
                                html.Label(
                                    "Operational thresholds (warning band):",
                                    style={"marginTop": "6px", "fontSize": "11px"},
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "6px"},
                                    children=[
                                        dcc.Input(
                                            id="threshold-low-input",
                                            type="number",
                                            placeholder="Low operational",
                                            style={"flex": 1},
                                        ),
                                        dcc.Input(
                                            id="threshold-high-input",
                                            type="number",
                                            placeholder="High operational",
                                            style={"flex": 1},
                                        ),
                                    ],
                                ),
                                html.Label(
                                    "Regulatory limits (permit/compliance):",
                                    style={"marginTop": "6px", "fontSize": "11px"},
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "6px"},
                                    children=[
                                        dcc.Input(
                                            id="threshold-low-limit-input",
                                            type="number",
                                            placeholder="Low limit",
                                            style={"flex": 1},
                                        ),
                                        dcc.Input(
                                            id="threshold-high-limit-input",
                                            type="number",
                                            placeholder="High limit",
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
    Output("threshold-alias-input", "value"),
    Output("threshold-units-input", "value"),
    Output("threshold-low-limit-input", "value"),
    Output("threshold-high-limit-input", "value"),
    Input("threshold-tag-dropdown", "value"),
)
def populate_threshold_inputs(tag):
    """Fill threshold inputs when tag changes."""
    th = load_thresholds()
    if not tag or tag not in th:
        return None, None, None, None, None, None
    entry = th[tag]
    return (
        entry.get("low_oper"),
        entry.get("high_oper"),
        entry.get("alias"),
        entry.get("units"),
        entry.get("low_limit"),
        entry.get("high_limit"),
    )


@app.callback(
    Output("threshold-save-status", "children"),
    Input("threshold-save-btn", "n_clicks"),
    State("threshold-tag-dropdown", "value"),
    State("threshold-low-input", "value"),
    State("threshold-high-input", "value"),
    State("threshold-alias-input", "value"),
    State("threshold-units-input", "value"),
    State("threshold-low-limit-input", "value"),
    State("threshold-high-limit-input", "value"),
    prevent_initial_call=True,
)
def save_thresholds_callback(
    n_clicks, tag, low_oper, high_oper, alias, units, low_limit, high_limit
):
    if not tag:
        return "Select a tag first."
    try:
        low_oper = float(low_oper) if low_oper not in (None, "") else None
    except ValueError:
        return "Operational low must be numeric."
    try:
        high_oper = float(high_oper) if high_oper not in (None, "") else None
    except ValueError:
        return "Operational high must be numeric."
    try:
        low_limit = float(low_limit) if low_limit not in (None, "") else None
    except ValueError:
        return "Regulatory low must be numeric."
    try:
        high_limit = float(high_limit) if high_limit not in (None, "") else None
    except ValueError:
        return "Regulatory high must be numeric."

    if (
        low_oper is not None
        and high_oper is not None
        and high_oper < low_oper
    ):
        low_oper, high_oper = high_oper, low_oper

    if (
        low_limit is not None
        and high_limit is not None
        and high_limit < low_limit
    ):
        low_limit, high_limit = high_limit, low_limit

    th = load_thresholds()
    old_entry = th.get(tag, {}) if isinstance(th.get(tag, {}), dict) else {}

    entry: Dict[str, object] = {}
    if alias:
        entry["alias"] = alias.strip()
    if units:
        entry["units"] = units.strip()
    if low_oper is not None:
        entry["low_oper"] = low_oper
    if high_oper is not None:
        entry["high_oper"] = high_oper
    if low_limit is not None:
        entry["low_limit"] = low_limit
    if high_limit is not None:
        entry["high_limit"] = high_limit

    th[tag] = entry
    save_thresholds(th)
    if entry != old_entry:
        log_config_change(
            field=f"Thresholds/Limits: {tag}",
            old_value=old_entry,
            new_value=entry,
            reason="Edited in web dashboard",
        )

    has_oper = (low_oper is not None) or (high_oper is not None)
    op_range = (
        f"operational [{low_oper}, {high_oper}]" if has_oper else "operational auto"
    )
    has_limit = (low_limit is not None) or (high_limit is not None)
    limit_range = f"limits [{low_limit}, {high_limit}]" if has_limit else "limits not set"
    return f"Saved {tag}: {op_range}; {limit_range}"


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
        events_df = load_env_events()
        thresholds = load_thresholds()
        config_version = compute_config_version_text()

        latest_by_tag, current_hour_avg, last_ts = extract_raw_stats(raw_df)
        last_hour_stats = extract_last_full_hour(hourly_df)

        tags_raw = set(latest_by_tag.keys())
        tags_hourly = set(last_hour_stats.keys())
        tags_cfg = set(load_configured_tags_from_settings())
        if not events_df.empty and "tag" in events_df.columns:
            tags_events = set(events_df["tag"].dropna().astype(str).tolist())
        else:
            tags_events = set()
        all_tags = sorted((tags_raw | tags_hourly | tags_cfg | tags_events), key=str)

        # Filter out numeric-looking tags that somehow slipped through
        all_tags = [t for t in all_tags if not looks_numeric_tag(str(t))]

        quality_stats = compute_quality_metrics(raw_df, events_df, all_tags)
        quality_card = build_quality_card(quality_stats)

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
            return [quality_card, empty_card], last_update
            return [empty_card], last_update, config_version

        cards = [quality_card]
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
