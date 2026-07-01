#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dash web dashboard for CIP Tag Poller

Reads:
    logs/raw_data_YYYY-MM-DD.csv   (latest day for current-hour averages)
    logs/hourly_averages.csv       (hourly aggregates)
    logs/rolling_12hr_averages.csv (rolling 12-hour averages)

Exposes a network dashboard with:
    - Simplified overview for CEMS O2/NOX/CO:
        * Gauge: Current Hourly Average
        * Rolling 12-hour Average summary
    - Color-coded gauges using thresholds.json (configurable in Thresholds tab).

Run:
    python CIPMonitor.py

Then open:
    http://<this_machine_ip>:8050

File layout
-----------
 1. Imports
 2. Configuration          -- file paths, CSV headers, EPA constants
 3. Styling                -- colors, card / button / gauge styles
 4. File & directory utils -- ensure_dir, ensure_csv, config-change log
 5. Export file helpers    -- lock, copy, filter, payload builders
 6. Report utilities       -- formatting, chart helpers, time-range math
 7. PDF report generation  -- emissions & incident reports
 8. Data loading           -- raw, hourly, rolling, events, exceedances,
                              thresholds, system health
 9. Settings & tag utils   -- alias maps, tag parsing, EPA settings
10. EPA Method 19 calcs    -- O2 correction, ppmv-to-lb/hr
11. Gauge helpers          -- classify_value, status_color, compute_gauge_range
12. DataFrame extraction   -- stats, last-hour, rolling-12hr, quality, compliance
13. Tag discovery          -- find_cems_tag, find_flow_tag, looks_numeric_tag
14. Gauge card builders    -- Flow, CEMS, Processing Time
15. Dashboard components   -- System Health, Data Quality, Compliance view
16. Dash app layout
17. Callbacks
18. Main entry point
"""

import os
import csv
import json
import hashlib
import getpass
import re
import shutil
import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pandas is anrequired for CIPMonitor. Install via 'pip install pandas'."
    ) from exc

try:
    from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, ctx
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Dash is required for CIPMonitor. Install via 'pip install dash'."
    ) from exc

try:
    import dash_daq as daq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "dash-daq is required for CIPMonitor. Install via 'pip install dash-daq'."
    ) from exc

# ============================================================================
#  CONFIGURATION -- file paths, CSV headers, EPA constants
# ============================================================================

LOG_DIR = "logs"
MINUTE_CSV = os.path.join(LOG_DIR, "minute_averages.csv")
HOURLY_CSV = os.path.join(LOG_DIR, "hourly_averages.csv")
ROLLING_12HR_CSV = os.path.join(LOG_DIR, "rolling_12hr_averages.csv")
LIVE_ROLLING_12HR_CSV = os.path.join(LOG_DIR, "rolling_12hr_live.csv")
RAW_CSV_PREFIX = "raw_data_"
THRESHOLDS_JSON = os.path.join(LOG_DIR, "thresholds.json")
SETTINGS_JSON = "settings.json"  # optional: to discover tags before any data
ENV_EVENTS_CSV = os.path.join(LOG_DIR, "env_events.csv")
EXCEEDANCES_CSV = os.path.join(LOG_DIR, "exceedances.csv")
CONFIG_CHANGES_CSV = os.path.join(LOG_DIR, "config_changes.csv")
SYSTEM_HEALTH_JSON = os.path.join(LOG_DIR, "system_health.json")
EXPORT_LOCK_FILENAME = ".cip_export.lock"
EXPORT_TMP_DIR = os.path.join(LOG_DIR, "exports")
CONFIG_CHANGE_HEADERS = [
    "timestamp",
    "user",
    "field",
    "old_value",
    "new_value",
    "reason",
]
MINUTE_AVG_HEADERS = [
    "minute_start",
    "minute_end",
    "tag",
    "alias",
    "avg_value",
    "avg_lb_hr",
    "sample_count",
]
RAW_DATA_HEADERS = [
    "timestamp",
    "date",
    "time",
    "tag",
    "alias",
    "value",
    "status",
    "qa_flag",
]
HOURLY_AVG_HEADERS = [
    "hour_start",
    "hour_end",
    "tag",
    "alias",
    "avg_value",
    "avg_lb_hr",
    "sample_count",
]
ROLLING_AVG_HEADERS = [
    "window_start",
    "window_end",
    "tag",
    "alias",
    "avg_value",
    "avg_lb_hr",
    "hours_count",
]

# EPA Method 19 constants used by the manual-calculation worksheet export.
EPA19_NOX_MOLECULAR_WEIGHT = 46.0   # NOx as NO2 (lb/lb-mol)
EPA19_CO_MOLECULAR_WEIGHT = 28.01   # CO (lb/lb-mol)
EPA19_MOLAR_VOLUME_DSCF = 385.3     # SCF/lb-mol at 68 F, 14.696 psia
EMISSION_TOTALS_HEADERS = [
    "pollutant",
    "period_label",
    "period_start",
    "period_end",
    "total_lb",
    "hours_with_data",
    "avg_lb_per_hr",
    "computation_basis",
]
NOX_CALC_HEADERS = [
    "period_start",
    "period_end",
    "nox_tag",
    "nox_alias",
    "nox_ppm",
    "flow_tag",
    "flow_alias",
    "flow_dscfm",
    "mw_no2_lb_per_lbmol",
    "molar_volume_dscf_per_lbmol",
    "calc_formula",
    "nox_lb_per_hr",
    "sample_count",
]

REFRESH_MS = 5000  # dashboard refresh interval (ms)

# Default regulatory permit limits (lb/hr) for CEMS pollutants.
# These are used as fallbacks when no high_limit is set in thresholds.json.
# They drive gauge scaling and the "Permit:" subtitle on lb/hr gauges.
DEFAULT_PERMIT_LIMITS_LB_HR = {
    "nox": 10.6,
    "co": 1.6,
}

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================================
#  STYLING -- colours, card / button / gauge container styles
# ============================================================================

COLOR_BG = "#0b0f14"
COLOR_SURFACE = "#151c25"
COLOR_SURFACE_ALT = "#1a2332"
COLOR_BORDER = "#253345"
COLOR_TEXT_PRIMARY = "#edf1f7"
COLOR_TEXT_MUTED = "#8b99ab"
COLOR_TEXT_SUBTLE = "#5e6e82"
COLOR_ACCENT = "#6c63ff"
COLOR_ACCENT_HOVER = "#7f78ff"
COLOR_ACCENT_BORDER = "#5a52e0"
COLOR_GOOD = "#34d399"
COLOR_WARNING = "#fbbf24"
COLOR_BAD = "#f87171"
COLOR_BUTTON_SECONDARY = "#374a60"
COLOR_BUTTON_SECONDARY_BORDER = "#2d3e52"
COLOR_BUTTON_TERTIARY = "#4a5d72"
COLOR_BUTTON_TERTIARY_BORDER = "#3e5065"
COLOR_TABLE_HEADER = "#1e2a38"
COLOR_TABLE_CELL = "#151c25"
COLOR_TABLE_CELL_ALT = "#192433"

CARD_STYLE = {
    "backgroundColor": COLOR_SURFACE,
    "borderRadius": "14px",
    "border": f"1px solid {COLOR_BORDER}",
    "padding": "18px 20px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "10px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.25)",
    "transition": "box-shadow 0.2s ease",
    "minWidth": "240px",
}

GAUGE_ROW_STYLE = {
    "display": "flex",
    "gap": "16px",
    "flexWrap": "wrap",
}

GAUGE_CONTAINER_STYLE = {
    "flex": "1",
    "textAlign": "center",
    "padding": "4px 0",
}

CARD_HEADER_STYLE = {
    "fontWeight": "700",
    "fontSize": "14px",
    "color": COLOR_TEXT_PRIMARY,
    "letterSpacing": "0.01em",
}

CARD_SUBTITLE_STYLE = {
    "fontSize": "11px",
    "color": COLOR_TEXT_MUTED,
    "letterSpacing": "0.02em",
}

CARD_FOOTER_STYLE = {
    "fontSize": "11px",
    "color": COLOR_TEXT_MUTED,
    "borderTop": f"1px solid {COLOR_BORDER}",
    "paddingTop": "8px",
    "marginTop": "2px",
}

EXPORT_SECTION_TITLE_STYLE = {
    "fontSize": "14px",
    "fontWeight": "700",
    "color": COLOR_TEXT_PRIMARY,
}

EXPORT_SECTION_HELP_STYLE = {
    "fontSize": "12px",
    "color": COLOR_TEXT_MUTED,
    "lineHeight": "1.5",
}

EXPORT_BUTTON_ROW_STYLE = {
    "display": "flex",
    "gap": "10px",
    "flexWrap": "wrap",
    "marginTop": "4px",
}

EXPORT_BUTTON_STYLE = {
    "backgroundColor": COLOR_ACCENT,
    "color": "white",
    "border": "none",
    "padding": "9px 16px",
    "borderRadius": "8px",
    "fontSize": "12px",
    "fontWeight": "600",
    "cursor": "pointer",
    "boxShadow": f"0 2px 8px {COLOR_ACCENT}33",
    "letterSpacing": "0.02em",
}

REPORT_BUTTON_STYLE = {
    "backgroundColor": COLOR_BUTTON_SECONDARY,
    "color": "white",
    "border": "none",
    "padding": "9px 16px",
    "borderRadius": "8px",
    "fontSize": "12px",
    "fontWeight": "600",
    "cursor": "pointer",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.2)",
}

INCIDENT_BUTTON_STYLE = {
    "backgroundColor": COLOR_BUTTON_TERTIARY,
    "color": "white",
    "border": "none",
    "padding": "9px 16px",
    "borderRadius": "8px",
    "fontSize": "12px",
    "fontWeight": "600",
    "cursor": "pointer",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.15)",
}

INPUT_STYLE = {
    "width": "100%",
    "backgroundColor": COLOR_SURFACE_ALT,
    "color": COLOR_TEXT_PRIMARY,
    "border": f"1px solid {COLOR_BORDER}",
    "borderRadius": "8px",
    "padding": "8px 10px",
    "fontSize": "12px",
    "outline": "none",
}

TAB_STYLE = {
    "backgroundColor": COLOR_SURFACE_ALT,
    "color": COLOR_TEXT_MUTED,
    "border": "none",
    "borderBottom": f"2px solid transparent",
    "padding": "10px 20px",
    "fontSize": "13px",
    "fontWeight": "500",
}

TAB_SELECTED_STYLE = {
    "backgroundColor": COLOR_SURFACE,
    "color": COLOR_TEXT_PRIMARY,
    "fontWeight": "700",
    "border": "none",
    "borderBottom": f"2px solid {COLOR_ACCENT}",
    "padding": "10px 20px",
    "fontSize": "13px",
}


# ============================================================================
#  FILE & DIRECTORY UTILITIES
# ============================================================================


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


# ============================================================================
#  EXPORT FILE HELPERS -- lock, copy, filter, payload builders
# ============================================================================


def ensure_export_csv(path: str, headers: List[str]) -> None:
    ensure_csv(path, headers)
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def _export_lock_path() -> str:
    return os.path.join(LOG_DIR, EXPORT_LOCK_FILENAME)


def _write_export_lock() -> None:
    ensure_dir(LOG_DIR)
    try:
        with open(_export_lock_path(), "w", encoding="utf-8") as f:
            f.write(f"export_started={datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"pid={os.getpid()}\n")
    except Exception:
        pass


def _clear_export_lock() -> None:
    try:
        os.remove(_export_lock_path())
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _copy_export_file(source_path: str, filename: str) -> Optional[str]:
    ensure_dir(EXPORT_TMP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{os.getpid()}_{filename}"
    dest_path = os.path.join(EXPORT_TMP_DIR, safe_name)
    try:
        shutil.copy2(source_path, dest_path)
        return dest_path
    except Exception:
        return None


def _copy_exports_for_sources(source_paths: List[str], filename_stub: str) -> List[str]:
    copied_paths: List[str] = []
    for source_path in source_paths:
        copied = _copy_export_file(source_path, f"{filename_stub}_{os.path.basename(source_path)}")
        if copied and os.path.exists(copied):
            copied_paths.append(copied)
    return copied_paths


def _filter_exported_csv_by_range(
    source_path: str,
    filename: str,
    headers: List[str],
    time_col: str,
    start: datetime,
    end: datetime,
) -> Optional[str]:
    copied_path = _copy_export_file(source_path, filename)
    if not copied_path or not os.path.exists(copied_path):
        return None

    try:
        df = pd.read_csv(copied_path)
    except Exception:
        return copied_path

    for col in headers:
        if col not in df.columns:
            df[col] = None

    try:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        filtered_df = _filter_time_range(df, time_col, start, end)
        export_df = filtered_df.reindex(columns=headers)
        export_df.to_csv(copied_path, index=False)
    except Exception:
        return copied_path
    return copied_path


def _get_raw_daily_paths_for_range(start: datetime, end: datetime) -> List[str]:
    if not os.path.isdir(LOG_DIR):
        return []
    start_date = start.date()
    end_date = end.date()
    # Include gzipped history: CIP.py compresses raw_data_*.csv -> .csv.gz after
    # 7 days, so a range older than a week is entirely gzip. Prefer the plain
    # .csv when both forms exist for the same date. pd.read_csv() decompresses
    # .gz transparently downstream (the copied temp file keeps its extension).
    by_date: Dict[datetime, Tuple[bool, str]] = {}
    for name in os.listdir(LOG_DIR):
        if not name.startswith(RAW_CSV_PREFIX):
            continue
        if name.endswith(".csv.gz"):
            date_part = name[len(RAW_CSV_PREFIX):-len(".csv.gz")]
            is_plain = False
        elif name.endswith(".csv"):
            date_part = name[len(RAW_CSV_PREFIX):-len(".csv")]
            is_plain = True
        else:
            continue
        try:
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except Exception:
            continue
        if file_date < start_date or file_date > end_date:
            continue
        key = datetime.combine(file_date, datetime.min.time())
        existing = by_date.get(key)
        if existing is None or (is_plain and not existing[0]):
            by_date[key] = (is_plain, os.path.join(LOG_DIR, name))
    return [path for _, (_, path) in sorted(by_date.items(), key=lambda item: item[0])]


def build_time_range_export_payload(
    source_path: str,
    headers: List[str],
    filename: str,
    time_col: str,
    range_key: str,
):
    ensure_export_csv(source_path, headers)
    if not os.path.exists(source_path):
        return None
    _, start, end = _get_report_range(range_key)
    _write_export_lock()
    try:
        export_copy = _filter_exported_csv_by_range(
            source_path,
            f"{range_key}_{filename}",
            headers,
            time_col,
            start,
            end,
        )
    finally:
        _clear_export_lock()
    if not export_copy or not os.path.exists(export_copy):
        return None
    return dcc.send_file(export_copy, filename=f"{range_key}_{filename}")


def build_raw_time_range_export_payload(range_key: str):
    _, start, end = _get_report_range(range_key)
    source_paths = _get_raw_daily_paths_for_range(start, end)
    if not source_paths:
        return None
    _write_export_lock()
    try:
        copied_sources = _copy_exports_for_sources(source_paths, f"{range_key}_raw")
        if not copied_sources:
            return None
        frames: List[pd.DataFrame] = []
        for path in copied_sources:
            try:
                df = pd.read_csv(path)
                for col in RAW_DATA_HEADERS:
                    if col not in df.columns:
                        df[col] = None
                frames.append(df.reindex(columns=RAW_DATA_HEADERS))
            except Exception:
                continue
        if not frames:
            return None
        export_df = pd.concat(frames, ignore_index=True)
        export_df["timestamp"] = pd.to_datetime(export_df["timestamp"], errors="coerce")
        export_df = _filter_time_range(export_df, "timestamp", start, end)
        output_path = os.path.join(
            EXPORT_TMP_DIR,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{range_key}_raw_data.csv",
        )
        export_df.reindex(columns=RAW_DATA_HEADERS).to_csv(output_path, index=False)
    finally:
        _clear_export_lock()
    if not os.path.exists(output_path):
        return None
    return dcc.send_file(output_path, filename=f"{range_key}_raw_data.csv")


def _resolve_nox_and_flow_tags(alias_map: Dict[str, str]) -> Tuple[str, str]:
    """Return (nox_tag, flow_tag) using EPA settings first, then alias patterns."""
    epa_settings = load_epa_settings()
    nox_tag = str(epa_settings.get("epa_nox_tag", "") or "")
    flow_tag = str(epa_settings.get("epa_flow_tag", "") or "")

    if not nox_tag or not flow_tag:
        nox_pattern = [re.compile(r"\bnox\b|no[_\s-]?x", re.IGNORECASE)]
        flow_pattern = [re.compile(r"\bflow\b|dscfm|scfm", re.IGNORECASE)]
        synthetic = pd.DataFrame({"tag": list(alias_map.keys())})
        if not nox_tag:
            nox_tag = _find_tag_by_patterns(synthetic, alias_map, nox_pattern)
        if not flow_tag:
            flow_tag = _find_tag_by_patterns(synthetic, alias_map, flow_pattern)
    return nox_tag, flow_tag


def _nox_calc_lb_per_hr(nox_ppm: Optional[float], flow_dscfm: Optional[float]) -> Optional[float]:
    if nox_ppm is None or flow_dscfm is None:
        return None
    try:
        return (
            float(nox_ppm)
            * float(flow_dscfm)
            * 60.0
            * EPA19_NOX_MOLECULAR_WEIGHT
            / (1_000_000.0 * EPA19_MOLAR_VOLUME_DSCF)
        )
    except (TypeError, ValueError):
        return None


def _build_nox_calc_rows_from_aggregated(
    source_path: str,
    start_col: str,
    end_col: str,
    count_col: str,
    nox_tag: str,
    flow_tag: str,
    alias_map: Dict[str, str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    if not os.path.exists(source_path):
        return pd.DataFrame(columns=NOX_CALC_HEADERS)

    df = pd.read_csv(source_path)
    if df.empty or "tag" not in df.columns:
        return pd.DataFrame(columns=NOX_CALC_HEADERS)

    df[end_col] = pd.to_datetime(df.get(end_col), errors="coerce")
    df[start_col] = pd.to_datetime(df.get(start_col), errors="coerce")
    df = _filter_time_range(df, end_col, start, end)
    if df.empty:
        return pd.DataFrame(columns=NOX_CALC_HEADERS)

    nox_df = df.loc[df["tag"] == nox_tag, [start_col, end_col, "avg_value", count_col]].copy()
    flow_df = df.loc[df["tag"] == flow_tag, [end_col, "avg_value"]].copy()
    nox_df = nox_df.rename(columns={"avg_value": "nox_ppm", count_col: "sample_count"})
    flow_df = flow_df.rename(columns={"avg_value": "flow_dscfm"})

    merged = nox_df.merge(flow_df, on=end_col, how="left")
    if merged.empty:
        return pd.DataFrame(columns=NOX_CALC_HEADERS)

    nox_alias = alias_map.get(nox_tag, "") if nox_tag else ""
    flow_alias = alias_map.get(flow_tag, "") if flow_tag else ""

    merged["nox_lb_per_hr"] = [
        _nox_calc_lb_per_hr(p, f) for p, f in zip(merged["nox_ppm"], merged["flow_dscfm"])
    ]
    merged["nox_tag"] = nox_tag
    merged["nox_alias"] = nox_alias
    merged["flow_tag"] = flow_tag
    merged["flow_alias"] = flow_alias
    merged["mw_no2_lb_per_lbmol"] = EPA19_NOX_MOLECULAR_WEIGHT
    merged["molar_volume_dscf_per_lbmol"] = EPA19_MOLAR_VOLUME_DSCF
    merged["calc_formula"] = "nox_ppm * flow_dscfm * 60 * 46.0 / (1e6 * 385.3)"
    merged = merged.rename(columns={start_col: "period_start", end_col: "period_end"})
    merged = merged.sort_values("period_end")
    return merged.reindex(columns=NOX_CALC_HEADERS)


def _build_nox_calc_rows_from_raw(
    nox_tag: str,
    flow_tag: str,
    alias_map: Dict[str, str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    source_paths = _get_raw_daily_paths_for_range(start, end)
    if not source_paths:
        return pd.DataFrame(columns=NOX_CALC_HEADERS)

    frames: List[pd.DataFrame] = []
    for path in source_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "tag" not in df.columns or "timestamp" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=NOX_CALC_HEADERS)

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = _filter_time_range(raw, "timestamp", start, end)
    if raw.empty:
        return pd.DataFrame(columns=NOX_CALC_HEADERS)

    if "status" in raw.columns:
        raw = raw.loc[raw["status"].astype(str).str.lower() == "success"]

    nox_rows = raw.loc[raw["tag"] == nox_tag, ["timestamp", "value"]].rename(columns={"value": "nox_ppm"})
    flow_rows = raw.loc[raw["tag"] == flow_tag, ["timestamp", "value"]].rename(columns={"value": "flow_dscfm"})
    if nox_rows.empty:
        return pd.DataFrame(columns=NOX_CALC_HEADERS)

    nox_rows["nox_ppm"] = pd.to_numeric(nox_rows["nox_ppm"], errors="coerce")
    flow_rows["flow_dscfm"] = pd.to_numeric(flow_rows["flow_dscfm"], errors="coerce")

    # Pair each NOx sample with the most recent flow sample at-or-before its timestamp.
    nox_rows = nox_rows.sort_values("timestamp")
    flow_rows = flow_rows.sort_values("timestamp")
    merged = pd.merge_asof(nox_rows, flow_rows, on="timestamp", direction="backward")

    merged["nox_lb_per_hr"] = [
        _nox_calc_lb_per_hr(p, f) for p, f in zip(merged["nox_ppm"], merged["flow_dscfm"])
    ]
    merged["period_start"] = merged["timestamp"]
    merged["period_end"] = merged["timestamp"]
    merged["nox_tag"] = nox_tag
    merged["nox_alias"] = alias_map.get(nox_tag, "") if nox_tag else ""
    merged["flow_tag"] = flow_tag
    merged["flow_alias"] = alias_map.get(flow_tag, "") if flow_tag else ""
    merged["mw_no2_lb_per_lbmol"] = EPA19_NOX_MOLECULAR_WEIGHT
    merged["molar_volume_dscf_per_lbmol"] = EPA19_MOLAR_VOLUME_DSCF
    merged["calc_formula"] = "nox_ppm * flow_dscfm * 60 * 46.0 / (1e6 * 385.3)"
    merged["sample_count"] = 1
    return merged.reindex(columns=NOX_CALC_HEADERS)


def build_nox_manual_calc_payload(granularity: str, range_key: str):
    """Build a NOx-only worksheet (ppm + flow + computed lb/hr) as CSV.

    granularity: "raw" | "minute" | "hourly". Raw pairs each NOx sample
    with the most recent flow sample (asof backward merge); minute/hourly
    use the corresponding aggregated CSV and join on the period end.
    """
    _, start, end = _get_report_range(range_key)
    alias_map = load_alias_map_from_settings()
    nox_tag, flow_tag = _resolve_nox_and_flow_tags(alias_map)
    if not nox_tag or not flow_tag:
        return None

    _write_export_lock()
    try:
        if granularity == "raw":
            rows = _build_nox_calc_rows_from_raw(
                nox_tag, flow_tag, alias_map, start, end
            )
        elif granularity == "minute":
            rows = _build_nox_calc_rows_from_aggregated(
                MINUTE_CSV,
                "minute_start",
                "minute_end",
                "sample_count",
                nox_tag,
                flow_tag,
                alias_map,
                start,
                end,
            )
        elif granularity == "hourly":
            rows = _build_nox_calc_rows_from_aggregated(
                HOURLY_CSV,
                "hour_start",
                "hour_end",
                "sample_count",
                nox_tag,
                flow_tag,
                alias_map,
                start,
                end,
            )
        else:
            return None

        if rows is None or rows.empty:
            return None

        ensure_dir(EXPORT_TMP_DIR)
        output_name = f"{range_key}_nox_manual_calc_{granularity}.csv"
        output_path = os.path.join(
            EXPORT_TMP_DIR,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{output_name}",
        )
        rows.to_csv(output_path, index=False)
    finally:
        _clear_export_lock()

    if not os.path.exists(output_path):
        return None
    return dcc.send_file(output_path, filename=output_name)


def _resolve_pollutant_and_flow_tags(
    alias_map: Dict[str, str],
) -> Tuple[str, str, str]:
    """Return (nox_tag, co_tag, flow_tag) from EPA settings with regex fallback."""
    epa_settings = load_epa_settings()
    nox_tag = str(epa_settings.get("epa_nox_tag", "") or "")
    co_tag = str(epa_settings.get("epa_co_tag", "") or "")
    flow_tag = str(epa_settings.get("epa_flow_tag", "") or "")

    if not nox_tag or not co_tag or not flow_tag:
        nox_pattern = [re.compile(r"\bnox\b|no[_\s-]?x", re.IGNORECASE)]
        co_pattern = [re.compile(r"\bco\b", re.IGNORECASE)]
        flow_pattern = [re.compile(r"\bflow\b|dscfm|scfm", re.IGNORECASE)]
        synthetic = pd.DataFrame({"tag": list(alias_map.keys())})
        if not nox_tag:
            nox_tag = _find_tag_by_patterns(synthetic, alias_map, nox_pattern)
        if not co_tag:
            co_tag = _find_tag_by_patterns(synthetic, alias_map, co_pattern)
        if not flow_tag:
            flow_tag = _find_tag_by_patterns(synthetic, alias_map, flow_pattern)
    return nox_tag, co_tag, flow_tag


def _emission_totals_row(
    pollutant: str,
    period_label: str,
    period_start: datetime,
    period_end: datetime,
    pollutant_tag: str,
    flow_tag: str,
    pollutant_mw: float,
    hourly_df: pd.DataFrame,
) -> Dict[str, object]:
    """Compute one totals row for a (pollutant, period) pair.

    Sums the per-hour lb/hr (recomputed from ppm * flow) over the period.
    No processing-state gating is applied -- this is the full emitted mass
    in any hour where the CEMS produced both ppm and flow data, which is
    what permit annual rollups care about.
    """
    period_window = _filter_time_range(hourly_df, "hour_start", period_start, period_end)
    rates = _hourly_lb_hr_recomputed(period_window, pollutant_tag, flow_tag, pollutant_mw)
    if rates.empty:
        total_lb = 0.0
        hours_with_data = 0
    else:
        total_lb = float(rates["lb_hr"].sum())
        hours_with_data = int(rates.shape[0])
    avg_lb_hr = (total_lb / hours_with_data) if hours_with_data else 0.0
    return {
        "pollutant": pollutant,
        "period_label": period_label,
        "period_start": period_start.isoformat(timespec="seconds"),
        "period_end": period_end.isoformat(timespec="seconds"),
        "total_lb": round(total_lb, 4),
        "hours_with_data": hours_with_data,
        "avg_lb_per_hr": round(avg_lb_hr, 6),
        "computation_basis": (
            "Sum of recomputed hourly lb/hr "
            "(ppm * dscfm * 60 * MW / (1e6 * 385.3))"
        ),
    }


def build_emission_totals_payload(range_key: str):
    """Emission totals CSV: selected range, calendar YTD, and trailing 12 months.

    For each pollutant (NOx, CO) emits three rows so a permit reviewer can
    see the requested window plus the two annual rollups in one file.
    Totals come from the simplified EPA Method 19 formula and are computed
    fresh from each hour's ppm and flow averages, not from the cached
    avg_lb_hr column on disk.
    """
    range_label, range_start, range_end = _get_report_range(range_key)
    alias_map = load_alias_map_from_settings()
    nox_tag, co_tag, flow_tag = _resolve_pollutant_and_flow_tags(alias_map)
    if not flow_tag or not (nox_tag or co_tag):
        return None

    hourly_df = load_hourly_stats()
    if hourly_df is None or hourly_df.empty:
        return None

    now = datetime.now()
    ytd_start = datetime(now.year, 1, 1)
    ytd_end = now
    trailing_start = now - timedelta(days=365)
    trailing_end = now

    periods: List[Tuple[str, datetime, datetime]] = [
        (f"Selected Range ({range_label})", range_start, range_end),
        (f"Year to Date {now.year}", ytd_start, ytd_end),
        ("Trailing 12 Months", trailing_start, trailing_end),
    ]

    rows: List[Dict[str, object]] = []
    pollutants: List[Tuple[str, str, float]] = []
    if nox_tag:
        pollutants.append(("NOx", nox_tag, EPA19_NOX_MOLECULAR_WEIGHT))
    if co_tag:
        pollutants.append(("CO", co_tag, EPA19_CO_MOLECULAR_WEIGHT))

    for pollutant_name, pollutant_tag, mw in pollutants:
        for label, p_start, p_end in periods:
            rows.append(
                _emission_totals_row(
                    pollutant_name,
                    label,
                    p_start,
                    p_end,
                    pollutant_tag,
                    flow_tag,
                    mw,
                    hourly_df,
                )
            )

    if not rows:
        return None

    out_df = pd.DataFrame(rows, columns=EMISSION_TOTALS_HEADERS)

    ensure_dir(EXPORT_TMP_DIR)
    output_name = f"{range_key}_emission_totals.csv"
    output_path = os.path.join(
        EXPORT_TMP_DIR,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{output_name}",
    )
    _write_export_lock()
    try:
        out_df.to_csv(output_path, index=False)
    finally:
        _clear_export_lock()

    if not os.path.exists(output_path):
        return None
    return dcc.send_file(output_path, filename=output_name)


def build_export_payload(path: str, headers: List[str], filename: str):
    ensure_export_csv(path, headers)
    if not os.path.exists(path):
        return None
    _write_export_lock()
    try:
        export_copy = _copy_export_file(path, filename)
    finally:
        _clear_export_lock()
    if not export_copy or not os.path.exists(export_copy):
        return None
    return dcc.send_file(export_copy, filename=filename)


# ============================================================================
#  REPORT UTILITIES -- formatting, chart helpers, time-range math
# ============================================================================


def _format_report_dt(value: Optional[datetime]) -> str:
    if not value or pd.isna(value):
        return "N/A"
    return value.strftime("%Y-%m-%d %H:%M")


def _add_limit_bands(ax: plt.Axes, limit_value: Optional[float]) -> None:
    if limit_value is None:
        return
    y_min, y_max = ax.get_ylim()
    if limit_value > y_max:
        y_max = limit_value * 1.1
    ax.set_ylim(y_min, y_max)

    green_end = min(limit_value * 0.8, y_max)
    yellow_end = min(limit_value, y_max)

    ax.axhspan(y_min, green_end, facecolor="#c8e6c9", alpha=0.25, zorder=0)
    ax.axhspan(green_end, yellow_end, facecolor="#fff9c4", alpha=0.3, zorder=0)
    if yellow_end < y_max:
        ax.axhspan(yellow_end, y_max, facecolor="#ffcdd2", alpha=0.25, zorder=0)


def _ensure_limit_visible(ax: plt.Axes, limit_value: Optional[float]) -> None:
    if limit_value is None:
        return
    y_min, y_max = ax.get_ylim()
    if limit_value > y_max:
        ax.set_ylim(y_min, limit_value * 1.1)


def _append_limit_lines(
    ax: plt.Axes,
    label_prefix: str,
    high_oper: Optional[float],
    high_limit: Optional[float],
    lines: List[matplotlib.lines.Line2D],
    labels: List[str],
    color: str = "#8b0000",
) -> None:
    if high_oper is not None:
        line = ax.axhline(
            high_oper,
            color=color,
            linewidth=1.2,
            linestyle="--",
        )
        lines.append(line)
        labels.append(f"{label_prefix} high oper")
    if high_limit is not None:
        line = ax.axhline(
            high_limit,
            color=color,
            linewidth=1.6,
        )
        lines.append(line)
        labels.append(f"{label_prefix} high limit")


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds:
        return "N/A"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60.0
    return f"{hours:.1f} hr"


def _filter_time_range(df: pd.DataFrame, time_col: str, start: datetime, end: datetime) -> pd.DataFrame:
    if df is None or df.empty or time_col not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    # Half-open [start, end): a row on the exact end boundary belongs to the
    # NEXT period, not this one. Combined with filtering hourly data on
    # hour_start (not hour_end), this keeps range membership consistent with how
    # weight totals (merged on hour_start) and CEMS uptime (hour_start) aggregate,
    # and stops the boundary hour from being double-counted across adjacent
    # periods (e.g. the last hour of a month leaking into the next month).
    mask = (df[time_col] >= start) & (df[time_col] < end)
    return df.loc[mask].copy()


def _find_tag_by_patterns(
    df: pd.DataFrame,
    alias_map: Dict[str, str],
    patterns: List[re.Pattern],
) -> str:
    if df is None or df.empty or "tag" not in df.columns:
        return ""
    for tag in df["tag"].dropna().unique():
        tag_str = str(tag)
        alias = alias_map.get(tag_str, "")
        combined = f"{tag_str} {alias}".casefold()
        if any(pattern.search(combined) for pattern in patterns):
            return tag_str
    return ""


def _display_name(tag: str, alias_map: Dict[str, str], fallback: str) -> str:
    if not tag:
        return fallback
    alias = alias_map.get(tag, "")
    if alias:
        return f"{alias} ({tag})"
    return tag


def _get_report_range(
    range_key: str,
    data_start: Optional[datetime] = None,
) -> Tuple[str, datetime, datetime]:
    now = datetime.now()
    if range_key == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        label = "Today"
        end = now
    elif range_key == "week":
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        label = "This Week"
        end = now
    elif range_key == "prev_month":
        first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_of_prev_month = first_of_this_month - timedelta(microseconds=1)
        start = last_of_prev_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        label = "Previous Month"
        end = last_of_prev_month
    elif range_key == "all_time":
        start = data_start or now.replace(hour=0, minute=0, second=0, microsecond=0)
        label = "All Time"
        end = now
    else:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        label = "This Month"
        end = now
    return label, start, end


def _get_report_data_start(*frames: pd.DataFrame) -> Optional[datetime]:
    earliest: Optional[datetime] = None
    for df in frames:
        if df is None or df.empty:
            continue
        for col in ("window_end", "hour_end"):
            if col in df.columns:
                series = pd.to_datetime(df[col], errors="coerce")
                if series.notna().any():
                    candidate = series.min()
                    if pd.notna(candidate):
                        ts = candidate.to_pydatetime()
                        if earliest is None or ts < earliest:
                            earliest = ts
    return earliest


def _is_system_failure(event_type: str) -> bool:
    normalized = str(event_type or "").strip().casefold()
    if not normalized:
        return False
    failure_terms = ("fail", "error", "down", "disconnect", "timeout", "offline")
    return any(term in normalized for term in failure_terms)


def _compute_avg_flow(hourly_range: pd.DataFrame, flow_tag: Optional[str]) -> Optional[float]:
    if not flow_tag or hourly_range is None or hourly_range.empty:
        return None
    flow_series = hourly_range.loc[hourly_range["tag"] == flow_tag]
    if flow_series.empty:
        return None
    values = flow_series.get("avg_value")
    if values is None:
        values = flow_series.get("avg_lb_hr")
    if values is None:
        return None
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _hourly_lb_hr_recomputed(
    hourly_range: pd.DataFrame,
    pollutant_tag: Optional[str],
    flow_tag: Optional[str],
    pollutant_mw: float,
) -> pd.DataFrame:
    """Per-hour lb/hr recomputed from ppm * flow, joined on hour_start.

    Rebuilds the EPA Method 19 mass rate from the underlying averages
    rather than trusting the cached avg_lb_hr column, so historical rows
    that were logged with the obsolete double-O2-correction formula
    contribute correct mass to range totals.

        lb/hr = ppm * Q_dscfm * 60 * MW / (1e6 * 385.3)

    Returns an empty DataFrame when inputs are missing.
    """
    if (
        not pollutant_tag
        or not flow_tag
        or hourly_range is None
        or hourly_range.empty
        or "tag" not in hourly_range.columns
        or "hour_start" not in hourly_range.columns
    ):
        return pd.DataFrame(columns=["hour_start", "lb_hr"])

    pollutant_rows = hourly_range.loc[
        hourly_range["tag"] == pollutant_tag, ["hour_start", "avg_value"]
    ].rename(columns={"avg_value": "ppm"})
    flow_rows = hourly_range.loc[
        hourly_range["tag"] == flow_tag, ["hour_start", "avg_value"]
    ].rename(columns={"avg_value": "flow_dscfm"})
    if pollutant_rows.empty or flow_rows.empty:
        return pd.DataFrame(columns=["hour_start", "lb_hr"])

    merged = pollutant_rows.merge(flow_rows, on="hour_start", how="inner")
    merged["ppm"] = pd.to_numeric(merged["ppm"], errors="coerce")
    merged["flow_dscfm"] = pd.to_numeric(merged["flow_dscfm"], errors="coerce")
    merged = merged.dropna(subset=["ppm", "flow_dscfm"])
    if merged.empty:
        return pd.DataFrame(columns=["hour_start", "lb_hr"])

    merged["lb_hr"] = (
        merged["ppm"]
        * merged["flow_dscfm"]
        * 60.0
        * pollutant_mw
        / (1_000_000.0 * EPA19_MOLAR_VOLUME_DSCF)
    )
    return merged[["hour_start", "lb_hr"]]


def _compute_total_weight(
    hourly_range: pd.DataFrame,
    pollutant_tag: Optional[str],
    flow_tag: Optional[str],
    pollutant_mw: float,
    processing_hour_starts: Optional[set] = None,
) -> Optional[float]:
    """Sum recomputed per-hour lb/hr to get total mass (lb) over the window.

    Each hour contributes ``lb/hr * 1 hour = lb``. The per-hour rate is
    recomputed from the hour's ppm and flow averages using the simplified
    EPA Method 19 formula (CEMS supplies O2-corrected, dry-basis values),
    so totals are correct even when the cached avg_lb_hr column on disk
    came from an older formula.

    When ``processing_hour_starts`` is provided, only hours whose
    hour_start is in that set are summed -- restricting the total to
    hours when the machine had any processing activity. The set must
    contain ``pandas.Timestamp`` objects.
    """
    rates = _hourly_lb_hr_recomputed(hourly_range, pollutant_tag, flow_tag, pollutant_mw)
    if rates.empty:
        return None
    if processing_hour_starts is not None:
        rates = rates.loc[rates["hour_start"].isin(processing_hour_starts)]
        if rates.empty:
            return None
    return float(rates["lb_hr"].sum())


# ============================================================================
#  PDF REPORT GENERATION -- emissions, incident & daily operations reports
# ============================================================================

# Shared report colour palette (light-background print-friendly)
_RPT = {
    "text": "#1b1f2a",
    "muted": "#3b4a5f",
    "subtle": "#5a6b82",
    "bg": "#f4f7fb",
    "panel": "#ffffff",
    "grid": "#d6e0ef",
    "header": "#dbe9f6",
    "alt_row": "#f1f6fc",
    "border": "#b7c7dd",
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d32f2f",
    "limit": "#6c6c6c",
    "bar_co": "#4a90d9",
    "bar_nox": "#f5a623",
    "bar_o2": "#7bc67e",
}


def generate_report_pdf(range_key: str) -> Optional[str]:
    rolling_df = load_rolling_12hr_stats()
    hourly_df = load_hourly_stats()
    data_start = _get_report_data_start(rolling_df, hourly_df)
    label, start, end = _get_report_range(range_key, data_start=data_start)
    alias_map = build_alias_lookup(
        pd.DataFrame(),
        hourly_df,
        rolling_df,
        load_alias_map_from_settings(),
    )
    thresholds = load_thresholds()
    epa_settings = load_epa_settings()

    rolling_range = _filter_time_range(rolling_df, "window_end", start, end)
    hourly_range = _filter_time_range(hourly_df, "hour_start", start, end)

    co_pattern = [re.compile(r"\bco\b|co[_\s-]", re.IGNORECASE)]
    nox_pattern = [re.compile(r"\bnox\b|no[_\s-]?x", re.IGNORECASE)]
    o2_pattern = [re.compile(r"\bo2\b|o2[_\s-]", re.IGNORECASE)]

    co_tag = _find_tag_by_patterns(rolling_range, alias_map, co_pattern)
    nox_tag = _find_tag_by_patterns(rolling_range, alias_map, nox_pattern)
    o2_tag = _find_tag_by_patterns(rolling_range, alias_map, o2_pattern)

    if not co_tag and not nox_tag and not o2_tag and not rolling_range.empty:
        co_tag = _find_tag_by_patterns(rolling_df, alias_map, co_pattern)
        nox_tag = _find_tag_by_patterns(rolling_df, alias_map, nox_pattern)
        o2_tag = _find_tag_by_patterns(rolling_df, alias_map, o2_pattern)

    # Also try hourly data for tag discovery (covers cases where rolling is empty)
    if not co_tag:
        co_tag = _find_tag_by_patterns(hourly_range, alias_map, co_pattern)
    if not nox_tag:
        nox_tag = _find_tag_by_patterns(hourly_range, alias_map, nox_pattern)
    if not o2_tag:
        o2_tag = _find_tag_by_patterns(hourly_range, alias_map, o2_pattern)

    co_series = rolling_range.loc[rolling_range["tag"] == co_tag] if co_tag else pd.DataFrame()
    nox_series = rolling_range.loc[rolling_range["tag"] == nox_tag] if nox_tag else pd.DataFrame()
    o2_series = rolling_range.loc[rolling_range["tag"] == o2_tag] if o2_tag else pd.DataFrame()

    co_entry = thresholds.get(co_tag, {}) if co_tag and isinstance(thresholds.get(co_tag, {}), dict) else {}
    nox_entry = thresholds.get(nox_tag, {}) if nox_tag and isinstance(thresholds.get(nox_tag, {}), dict) else {}

    co_high_oper = _to_float(co_entry.get("high_oper"))
    nox_high_oper = _to_float(nox_entry.get("high_oper"))

    tags = sorted(set(hourly_range.get("tag", pd.Series(dtype=str)).dropna().astype(str)))
    flow_tag = str(epa_settings.get("epa_flow_tag", "") or "")
    if not flow_tag:
        flow_tag = find_flow_tag(tags, thresholds, alias_map)
    flow_avg = _compute_avg_flow(hourly_range, flow_tag)
    flow_label = _display_name(flow_tag, alias_map, "Flow")
    machine_state_tag = load_machine_state_tag_from_settings()

    # Precise processing-time calculation from raw data (sub-minute granularity)
    days_span = max(1, (datetime.now().date() - start.date()).days + 2)
    raw_history = load_raw_history(max_days=min(days_span, 400))
    processing_stats = compute_processing_time_range(
        raw_history, machine_state_tag, start, end,
    )
    processing_hour_starts = processing_stats["hour_starts_with_any"]

    # Secondary: weight-decrease-based conveyor run time (always computed
    # when a weight tag is configured, regardless of primary detection method)
    feed_settings = load_feed_settings()
    weight_tag_for_metric = feed_settings.get("weight_tag", "")
    weight_run_stats = compute_weight_decrease_time_range(
        raw_history, weight_tag_for_metric, start, end,
    ) if weight_tag_for_metric else None
    if weight_run_stats:
        weight_run_stats["weight_units"] = feed_settings.get("weight_units", "")

    co_weight = _compute_total_weight(
        hourly_range, co_tag, flow_tag, EPA19_CO_MOLECULAR_WEIGHT, processing_hour_starts
    )
    nox_weight = _compute_total_weight(
        hourly_range, nox_tag, flow_tag, EPA19_NOX_MOLECULAR_WEIGHT, processing_hour_starts
    )

    # CEMS uptime for the reporting period
    cems_tags = [t for t in [co_tag, nox_tag, o2_tag] if t]
    uptime = compute_cems_uptime(hourly_df, machine_state_tag, cems_tags, start, end)
    uptime_pct = uptime.get("uptime_pct", 0.0)
    op_hours = uptime.get("operating_hours", 0)
    valid_hours = uptime.get("valid_cems_hours", 0)
    total_report_hours = float(uptime.get("total_hours", 0)) or (
        (end - start).total_seconds() / 3600.0
    )

    R = _RPT  # shorthand

    ensure_dir(EXPORT_TMP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cip_report_{range_key}_{timestamp}.pdf"
    report_path = os.path.join(EXPORT_TMP_DIR, filename)

    with PdfPages(report_path) as pdf:
        # ---- Page 1: Rolling 12-hr chart + summary table ----
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(R["bg"])
        gs = fig.add_gridspec(
            3, 1,
            height_ratios=[0.18, 0.55, 0.27],
            left=0.07, right=0.78, top=0.95, bottom=0.06,
            hspace=0.45,
        )

        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.axis("off")
        ax_title.text(0.0, 0.75, f"CIP Emissions Report - {label}",
                      fontsize=18, fontweight="bold", color=R["text"])
        ax_title.text(0.0, 0.4,
                      f"Reporting Window: {_format_report_dt(start)} to {_format_report_dt(end)}",
                      fontsize=11, color=R["muted"])
        ax_title.text(0.0, 0.1,
                      "Rolling 12-hour averages for CO/NOx (lb/hr) with %O2 trend overlay.",
                      fontsize=9.5, color=R["subtle"])

        ax_chart = fig.add_subplot(gs[1, 0])
        ax_chart.set_title("Rolling 12-Hour Averages", fontsize=12, color=R["text"])
        ax_chart.set_ylabel("lb/hr", color=R["text"])
        ax_chart.set_facecolor(R["panel"])
        ax_chart.grid(True, linestyle="--", color=R["grid"], alpha=0.6)
        ax_chart.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d\n%H:%M"))

        lines = []
        labels = []

        if not co_series.empty:
            line, = ax_chart.plot(co_series["window_end"], co_series["avg_lb_hr"],
                                 color=R["blue"], linewidth=2, linestyle="-")
            lines.append(line)
            labels.append(f"CO 12-hr avg ({_display_name(co_tag, alias_map, 'CO')})")

        if not nox_series.empty:
            line, = ax_chart.plot(nox_series["window_end"], nox_series["avg_lb_hr"],
                                 color=R["orange"], linewidth=2, linestyle="--")
            lines.append(line)
            labels.append(f"NOx 12-hr avg ({_display_name(nox_tag, alias_map, 'NOx')})")

        ax_o2 = ax_chart.twinx()
        ax_o2.set_ylabel("%O2", color=R["text"])
        ax_o2.set_facecolor("none")
        if not o2_series.empty:
            line, = ax_o2.plot(o2_series["window_end"], o2_series["avg_value"],
                               color=R["green"], linestyle=":", linewidth=2)
            lines.append(line)
            labels.append(f"O2 % ({_display_name(o2_tag, alias_map, 'O2')})")

        if co_high_oper is not None:
            _ensure_limit_visible(ax_chart, co_high_oper)
            _append_limit_lines(ax_chart, "CO", co_high_oper, None, lines, labels, color=R["limit"])
        if nox_high_oper is not None:
            _ensure_limit_visible(ax_chart, nox_high_oper)
            _append_limit_lines(ax_chart, "NOx", nox_high_oper, None, lines, labels, color=R["limit"])

        if lines:
            ax_chart.legend(lines, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                            fontsize=8, frameon=False, borderaxespad=0.0)
        else:
            ax_chart.text(0.5, 0.5, "No rolling 12-hour data available for the selected window.",
                          ha="center", va="center", fontsize=11, color=R["muted"],
                          transform=ax_chart.transAxes)

        ax_table = fig.add_subplot(gs[2, 0])
        ax_table.axis("off")

        report_avail_pct = (
            (valid_hours / total_report_hours * 100.0)
            if total_report_hours > 0 else 0.0
        )
        if total_report_hours > 0:
            uptime_text = (
                f"{report_avail_pct:.1f}% of report "
                f"({valid_hours}/{int(round(total_report_hours))} hrs)"
            )
            if op_hours > 0:
                uptime_text += f"; {uptime_pct:.1f}% of operating"
        else:
            uptime_text = "N/A"

        hours_processed_precise = float(processing_stats.get("total_hours", 0.0))
        capped_gaps = int(processing_stats.get("capped_gap_count", 0))
        proc_pct = (
            (hours_processed_precise / total_report_hours * 100.0)
            if total_report_hours > 0 else 0.0
        )
        hours_text = (
            f"{hours_processed_precise:.2f} / {total_report_hours:.2f} hrs "
            f"({proc_pct:.1f}%)"
        )
        if capped_gaps > 0:
            hours_text += f"  ({capped_gaps} gap(s) capped)"

        summary_rows = [
            [f"Average Flow Rate ({flow_label})", f"{flow_avg:.2f}" if flow_avg is not None else "N/A"],
            ["CO Total Weight (lb)", f"{co_weight:.2f}" if co_weight is not None else "N/A"],
            ["NOx Total Weight (lb)", f"{nox_weight:.2f}" if nox_weight is not None else "N/A"],
            ["Hours Processed", hours_text],
            ["CEMS Data Availability", uptime_text],
        ]

        if weight_run_stats and weight_run_stats.get("sample_count", 0) > 0:
            run_hrs = float(weight_run_stats.get("total_hours", 0.0))
            avg_rate = weight_run_stats.get("avg_feed_rate_per_min")
            w_units = weight_run_stats.get("weight_units") or "units"
            run_text = f"{run_hrs:.2f} hrs"
            if avg_rate is not None and avg_rate > 0:
                run_text += f"  (avg {avg_rate:.1f} {w_units}/min)"
            summary_rows.append(["Conveyor Run Time (weight)", run_text])

        tbl = ax_table.table(cellText=summary_rows, colLabels=["Metric", "Value"],
                             cellLoc="left", colLoc="left", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1, 1.35)
        for (row, _col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color=R["text"])
                cell.set_facecolor(R["header"])
            else:
                cell.set_facecolor(R["alt_row"])
                cell.set_edgecolor(R["border"])

        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: Hourly averages bar chart ----
        _add_hourly_chart_page(pdf, hourly_range, co_tag, nox_tag, o2_tag,
                               alias_map, thresholds, label, start, end)

    return report_path


def _add_hourly_chart_page(
    pdf: PdfPages,
    hourly_range: pd.DataFrame,
    co_tag: str,
    nox_tag: str,
    o2_tag: str,
    alias_map: Dict[str, str],
    thresholds: Dict[str, Dict[str, object]],
    label: str,
    start: datetime,
    end: datetime,
) -> None:
    """Render a second PDF page showing hourly average bar charts for CO, NOx, and O2."""
    R = _RPT

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(R["bg"])

    # Title area
    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax_title.axis("off")
    ax_title.text(0.0, 0.7, f"Hourly Averages - {label}", fontsize=16, fontweight="bold", color=R["text"])
    ax_title.text(0.0, 0.0,
                  f"{_format_report_dt(start)} to {_format_report_dt(end)}",
                  fontsize=10, color=R["muted"])

    plot_specs = []
    if co_tag:
        plot_specs.append(("CO", co_tag, "avg_lb_hr", "lb/hr", R["bar_co"]))
    if nox_tag:
        plot_specs.append(("NOx", nox_tag, "avg_lb_hr", "lb/hr", R["bar_nox"]))
    if o2_tag:
        plot_specs.append(("O2", o2_tag, "avg_value", "%", R["bar_o2"]))

    n_plots = max(len(plot_specs), 1)
    # Reserve room at top for the title block and at the bottom so rotated
    # x-tick labels are never clipped by the page margin.
    top_reserve = 0.10
    bottom_reserve = 0.08
    plot_gap = 0.04
    usable = 1.0 - top_reserve - bottom_reserve
    per_plot = (usable - (n_plots - 1) * plot_gap) / n_plots

    for idx, (name, tag, value_col, unit, color) in enumerate(plot_specs):
        bottom = (
            1.0 - top_reserve - (idx + 1) * per_plot - idx * plot_gap
        )
        ax = fig.add_axes([0.08, bottom, 0.86, per_plot])
        ax.set_facecolor(R["panel"])
        ax.grid(True, axis="y", linestyle="--", color=R["grid"], alpha=0.5)

        tag_data = hourly_range.loc[hourly_range["tag"] == tag].copy() if not hourly_range.empty else pd.DataFrame()

        if tag_data.empty or value_col not in tag_data.columns:
            ax.text(0.5, 0.5, f"No hourly data for {_display_name(tag, alias_map, name)}",
                    ha="center", va="center", fontsize=10, color=R["muted"], transform=ax.transAxes)
            ax.set_title(f"{name} ({_display_name(tag, alias_map, name)})",
                         fontsize=11, color=R["text"], loc="left")
            continue

        tag_data = tag_data.sort_values("hour_start")
        hours = tag_data["hour_start"]
        values = pd.to_numeric(tag_data[value_col], errors="coerce").fillna(0)

        ax.bar(hours, values, width=timedelta(hours=0.8), color=color, alpha=0.85, edgecolor="none")
        ax.set_ylabel(unit, fontsize=9, color=R["text"])
        ax.set_title(f"{name} Hourly Average ({_display_name(tag, alias_map, name)})",
                     fontsize=11, color=R["text"], loc="left")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=8)

        # Add limit lines
        entry = thresholds.get(tag, {}) if isinstance(thresholds.get(tag, {}), dict) else {}
        high_oper = _to_float(entry.get("high_oper"))
        high_limit = _to_float(entry.get("high_limit"))
        if high_oper is not None:
            ax.axhline(high_oper, color=R["limit"], linewidth=1, linestyle="--", label="Oper limit")
            _ensure_limit_visible(ax, high_oper)
        if high_limit is not None:
            ax.axhline(high_limit, color=R["red"], linewidth=1.2, linestyle="-", label="Reg limit")
            _ensure_limit_visible(ax, high_limit)
        if high_oper is not None or high_limit is not None:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    if not plot_specs:
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.6])
        ax.axis("off")
        ax.text(0.5, 0.5, "No CEMS tags found for hourly chart.",
                ha="center", va="center", fontsize=12, color=R["muted"])

    pdf.savefig(fig)
    plt.close(fig)


def generate_daily_ops_pdf(range_key: str) -> Optional[str]:
    """Generate a Daily Operations PDF report.

    Page 1 -- 24-hour hourly-average bar charts (CO, NOx, O2) with limit lines
    Page 2 -- Operational summary (hours processed, flow, weight totals) + hourly data table
    """
    hourly_df = load_hourly_stats()
    rolling_df = load_rolling_12hr_stats()
    raw_df = load_latest_raw_df()
    data_start = _get_report_data_start(rolling_df, hourly_df)
    label, start, end = _get_report_range(range_key, data_start=data_start)
    alias_map = build_alias_lookup(raw_df, hourly_df, rolling_df, load_alias_map_from_settings())
    thresholds = load_thresholds()
    epa_settings = load_epa_settings()

    hourly_range = _filter_time_range(hourly_df, "hour_start", start, end)
    rolling_range = _filter_time_range(rolling_df, "window_end", start, end)

    co_pattern = [re.compile(r"\bco\b|co[_\s-]", re.IGNORECASE)]
    nox_pattern = [re.compile(r"\bnox\b|no[_\s-]?x", re.IGNORECASE)]
    o2_pattern = [re.compile(r"\bo2\b|o2[_\s-]", re.IGNORECASE)]

    co_tag = (_find_tag_by_patterns(hourly_range, alias_map, co_pattern)
              or _find_tag_by_patterns(rolling_range, alias_map, co_pattern))
    nox_tag = (_find_tag_by_patterns(hourly_range, alias_map, nox_pattern)
               or _find_tag_by_patterns(rolling_range, alias_map, nox_pattern))
    o2_tag = (_find_tag_by_patterns(hourly_range, alias_map, o2_pattern)
              or _find_tag_by_patterns(rolling_range, alias_map, o2_pattern))

    tags = sorted(set(hourly_range.get("tag", pd.Series(dtype=str)).dropna().astype(str)))
    flow_tag = str(epa_settings.get("epa_flow_tag", "") or "")
    if not flow_tag:
        flow_tag = find_flow_tag(tags, thresholds, alias_map)

    machine_state_tag = load_machine_state_tag_from_settings()
    cems_tags = [t for t in [co_tag, nox_tag, o2_tag] if t]
    uptime = compute_cems_uptime(hourly_df, machine_state_tag, cems_tags, start, end)

    R = _RPT
    ensure_dir(EXPORT_TMP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cip_daily_ops_{range_key}_{timestamp}.pdf"
    report_path = os.path.join(EXPORT_TMP_DIR, filename)

    with PdfPages(report_path) as pdf:
        # ---- Page 1: Hourly bar charts ----
        _add_hourly_chart_page(pdf, hourly_range, co_tag, nox_tag, o2_tag,
                               alias_map, thresholds, label, start, end)

        # ---- Page 2: Operational summary + data table ----
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(R["bg"])

        # Title
        ax_hdr = fig.add_axes([0.05, 0.92, 0.9, 0.06])
        ax_hdr.axis("off")
        ax_hdr.text(0.0, 0.7, f"Operational Summary - {label}",
                    fontsize=16, fontweight="bold", color=R["text"])
        ax_hdr.text(0.0, 0.0,
                    f"{_format_report_dt(start)} to {_format_report_dt(end)}",
                    fontsize=10, color=R["muted"])

        # --- Left: CEMS uptime + operations stats ---
        ax_stats = fig.add_axes([0.05, 0.40, 0.42, 0.46])
        ax_stats.axis("off")

        op_hrs = int(uptime.get("operating_hours", 0))
        valid_hrs = int(uptime.get("valid_cems_hours", 0))
        missing_hrs = int(uptime.get("missing_hours", 0))
        uptime_pct = float(uptime.get("uptime_pct", 0))
        total_report_hours = float(uptime.get("total_hours", 0)) or (
            (end - start).total_seconds() / 3600.0
        )
        report_avail_pct = (
            (valid_hrs / total_report_hours * 100.0)
            if total_report_hours > 0 else 0.0
        )
        flow_avg = _compute_avg_flow(hourly_range, flow_tag)

        # Precise processing-time calculation from raw data (sub-minute granularity)
        days_span = max(1, (datetime.now().date() - start.date()).days + 2)
        raw_history = load_raw_history(max_days=min(days_span, 400))
        processing_stats = compute_processing_time_range(
            raw_history, machine_state_tag, start, end,
        )
        processing_hour_starts = processing_stats["hour_starts_with_any"]
        hours_processed_precise = float(processing_stats.get("total_hours", 0.0))
        capped_gaps = int(processing_stats.get("capped_gap_count", 0))

        co_weight = _compute_total_weight(
            hourly_range, co_tag, flow_tag, EPA19_CO_MOLECULAR_WEIGHT, processing_hour_starts
        )
        nox_weight = _compute_total_weight(
            hourly_range, nox_tag, flow_tag, EPA19_NOX_MOLECULAR_WEIGHT, processing_hour_starts
        )

        hours_text = f"{hours_processed_precise:.2f} hrs"
        if capped_gaps > 0:
            hours_text += f" ({capped_gaps} capped)"

        # Secondary: weight-decrease-based conveyor run time
        feed_settings = load_feed_settings()
        weight_tag_for_metric = feed_settings.get("weight_tag", "")
        weight_run_stats = compute_weight_decrease_time_range(
            raw_history, weight_tag_for_metric, start, end,
        ) if weight_tag_for_metric else None

        flow_label = _display_name(flow_tag, alias_map, "Flow")

        # CEMS Availability rows intentionally omitted from the operational
        # summary (availability is reported in the incident report instead).
        stats_rows = [
            ["Hours Processed", hours_text],
            [f"Avg Flow ({flow_label})", f"{flow_avg:.1f}" if flow_avg is not None else "N/A"],
            ["CO Total Weight (lb)", f"{co_weight:.2f}" if co_weight is not None else "N/A"],
            ["NOx Total Weight (lb)", f"{nox_weight:.2f}" if nox_weight is not None else "N/A"],
        ]

        if weight_run_stats and weight_run_stats.get("sample_count", 0) > 0:
            run_hrs = float(weight_run_stats.get("total_hours", 0.0))
            avg_rate = weight_run_stats.get("avg_feed_rate_per_min")
            w_units = feed_settings.get("weight_units", "") or "units"
            run_text = f"{run_hrs:.2f} hrs"
            if avg_rate is not None and avg_rate > 0:
                run_text += f" (avg {avg_rate:.1f} {w_units}/min)"
            stats_rows.append(["Conveyor Run (weight)", run_text])

        tbl = ax_stats.table(cellText=stats_rows, colLabels=["Metric", "Value"],
                             cellLoc="left", colLoc="left", loc="upper left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.3)
        for (row, _c), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color=R["text"])
                cell.set_facecolor(R["header"])
            else:
                cell.set_facecolor(R["alt_row"])
                cell.set_edgecolor(R["border"])

        # --- Right: Hourly data table (last 24 rows) ---
        ax_data = fig.add_axes([0.52, 0.40, 0.46, 0.46])
        ax_data.axis("off")
        ax_data.set_title("Hourly Averages (most recent)", fontsize=11,
                          color=R["text"], loc="left", pad=8)

        # Build a pivot of the last hours
        data_rows: List[List[str]] = []
        data_cols = ["Hour", "CO (lb/hr)", "NOx (lb/hr)", "O2 (%)"]
        if not hourly_range.empty:
            hr_sorted = hourly_range.sort_values("hour_start")
            unique_hours = sorted(hr_sorted["hour_start"].dropna().unique())[-24:]
            for h in unique_hours:
                hour_data = hr_sorted.loc[hr_sorted["hour_start"] == h]
                try:
                    h_label = pd.Timestamp(h).strftime("%m-%d %H:%M")
                except Exception:
                    h_label = str(h)
                co_val = ""
                nox_val = ""
                o2_val = ""
                if co_tag:
                    co_row = hour_data.loc[hour_data["tag"] == co_tag]
                    if not co_row.empty:
                        v = pd.to_numeric(co_row.iloc[0].get("avg_lb_hr"), errors="coerce")
                        co_val = f"{v:.2f}" if pd.notna(v) else ""
                if nox_tag:
                    nox_row = hour_data.loc[hour_data["tag"] == nox_tag]
                    if not nox_row.empty:
                        v = pd.to_numeric(nox_row.iloc[0].get("avg_lb_hr"), errors="coerce")
                        nox_val = f"{v:.2f}" if pd.notna(v) else ""
                if o2_tag:
                    o2_row = hour_data.loc[hour_data["tag"] == o2_tag]
                    if not o2_row.empty:
                        v = pd.to_numeric(o2_row.iloc[0].get("avg_value"), errors="coerce")
                        o2_val = f"{v:.2f}" if pd.notna(v) else ""
                data_rows.append([h_label, co_val, nox_val, o2_val])

        if data_rows:
            dtbl = ax_data.table(cellText=data_rows, colLabels=data_cols,
                                 cellLoc="center", colLoc="center", loc="upper left")
            dtbl.auto_set_font_size(False)
            dtbl.set_fontsize(7.5)
            dtbl.scale(1, 1.1)
            for (row, _c), cell in dtbl.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight="bold", color=R["text"])
                    cell.set_facecolor(R["header"])
                else:
                    cell.set_facecolor(R["alt_row"] if row % 2 == 0 else R["panel"])
                    cell.set_edgecolor(R["border"])
        else:
            ax_data.text(0.5, 0.5, "No hourly data available.",
                         ha="center", va="center", fontsize=10, color=R["muted"],
                         transform=ax_data.transAxes)

        # --- Bottom: Compliance note ---
        ax_footer = fig.add_axes([0.05, 0.02, 0.9, 0.08])
        ax_footer.axis("off")
        ax_footer.text(
            0.0, 0.8,
            "Hours Processed = time the machine was in Processing state (sub-minute "
            "precision from raw samples). Weight totals sum per-hour lb/hr over the "
            "period. Hours are attributed to a period by their start time.",
            fontsize=8.5, color=R["subtle"], style="italic",
        )
        ax_footer.text(
            0.0, 0.2,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=8, color=R["subtle"],
        )

        pdf.savefig(fig)
        plt.close(fig)

    return report_path


def build_daily_ops_payload(range_key: str):
    _write_export_lock()
    try:
        report_path = generate_daily_ops_pdf(range_key)
    finally:
        _clear_export_lock()
    if not report_path or not os.path.exists(report_path):
        return None
    filename = os.path.basename(report_path)
    return dcc.send_file(report_path, filename=filename)


def generate_incident_report_pdf(range_key: str) -> Optional[str]:
    raw_df = load_latest_raw_df()
    hourly_df = load_hourly_stats()
    rolling_df = load_rolling_12hr_stats()
    data_start = _get_report_data_start(rolling_df, hourly_df)
    label, start, end = _get_report_range(range_key, data_start=data_start)

    exceedances_df = load_exceedances()
    env_events_df = load_env_events()
    system_health = load_system_health()

    exceed_range = _filter_time_range(exceedances_df, "start_time", start, end)
    system_events = _filter_time_range(env_events_df, "timestamp", start, end)
    if not system_events.empty and "event_type" in system_events.columns:
        system_events = system_events[
            system_events["event_type"].astype(str).apply(_is_system_failure)
        ]

    exceed_count = int(exceed_range.shape[0]) if not exceed_range.empty else 0
    exceed_minutes = (
        float(exceed_range["duration_sec"].fillna(0).astype(float).sum()) / 60.0
        if not exceed_range.empty and "duration_sec" in exceed_range.columns
        else 0.0
    )
    failure_count = int(system_events.shape[0]) if not system_events.empty else 0
    failure_minutes = (
        float(system_events["duration_sec"].fillna(0).astype(float).sum()) / 60.0
        if not system_events.empty and "duration_sec" in system_events.columns
        else 0.0
    )

    threshold_summary = compute_compliance_summary(
        raw_df, load_thresholds(), load_exceedances()
    )

    # CEMS uptime for the reporting period
    machine_state_tag = load_machine_state_tag_from_settings()
    alias_map = build_alias_lookup(raw_df, hourly_df, rolling_df, load_alias_map_from_settings())
    thresholds = load_thresholds()
    tags = sorted(set(hourly_df.get("tag", pd.Series(dtype=str)).dropna().astype(str)))
    cems_tags = []
    for metric in ("o2", "nox", "co"):
        t = find_cems_tag(metric, tags, thresholds, alias_map)
        if t:
            cems_tags.append(t)
    uptime = compute_cems_uptime(hourly_df, machine_state_tag, cems_tags, start, end)
    uptime_pct = float(uptime.get("uptime_pct", 0))
    op_hrs = int(uptime.get("operating_hours", 0))
    valid_hrs = int(uptime.get("valid_cems_hours", 0))
    total_report_hours = float(uptime.get("total_hours", 0)) or (
        (end - start).total_seconds() / 3600.0
    )
    report_avail_pct = (
        (valid_hrs / total_report_hours * 100.0)
        if total_report_hours > 0 else 0.0
    )

    R = _RPT

    ensure_dir(EXPORT_TMP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cip_incident_report_{range_key}_{timestamp}.pdf"
    report_path = os.path.join(EXPORT_TMP_DIR, filename)

    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(R["bg"])
        gs = fig.add_gridspec(
            3, 1,
            height_ratios=[0.18, 0.42, 0.40],
            left=0.06, right=0.96, top=0.95, bottom=0.05,
            hspace=0.55,
        )

        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.axis("off")
        ax_title.text(0.0, 0.75, f"CIP Exceedances & System Failures - {label}",
                      fontsize=17, fontweight="bold", color=R["text"])
        ax_title.text(0.0, 0.4,
                      f"Reporting Window: {_format_report_dt(start)} to {_format_report_dt(end)}",
                      fontsize=11, color=R["muted"])
        ax_title.text(0.0, 0.1,
                      "Exceedances reflect regulatory limit violations; "
                      "system failures summarize health and data events.",
                      fontsize=9.5, color=R["subtle"])

        ax_summary = fig.add_subplot(gs[1, 0])
        ax_summary.axis("off")

        if total_report_hours > 0:
            uptime_text = (
                f"{report_avail_pct:.1f}% of report "
                f"({valid_hrs}/{int(round(total_report_hours))} hrs)"
            )
            if op_hrs > 0:
                uptime_text += f"; {uptime_pct:.1f}% of operating"
        else:
            uptime_text = "N/A"
        summary_rows = [
            ["Exceedance events", str(exceed_count)],
            ["Exceedance duration", f"{exceed_minutes:.1f} min"],
            ["System failure events", str(failure_count)],
            ["Failure duration", f"{failure_minutes:.1f} min"],
            ["Within limits (last 24h)", f"{threshold_summary.get('pct_24h', 0.0):.1f}%"],
            ["Within limits (last 30d)", f"{threshold_summary.get('pct_30d', 0.0):.1f}%"],
            ["CEMS Data Availability", uptime_text],
            ["System health status", str(system_health.get("status", "Unknown"))],
            ["System health detail", str(system_health.get("status_reason", ""))],
        ]
        summary_table = ax_summary.table(
            cellText=summary_rows, colLabels=["Metric", "Value"],
            cellLoc="left", colLoc="left", loc="center",
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(10)
        summary_table.scale(1, 1.25)
        for (row, _c), cell in summary_table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color=R["text"])
                cell.set_facecolor(R["header"])
            else:
                cell.set_facecolor(R["alt_row"])
                cell.set_edgecolor(R["border"])

        ax_tables = fig.add_subplot(gs[2, 0])
        ax_tables.axis("off")

        def _prepare_exceed_rows(df: pd.DataFrame) -> List[List[str]]:
            if df.empty:
                return [["No exceedance events in this window.", "", "", ""]]
            rows: List[List[str]] = []
            for _, row in df.sort_values("start_time", ascending=False).head(8).iterrows():
                rows.append([
                    str(row.get("tag", "")),
                    _format_report_dt(pd.to_datetime(row.get("start_time"), errors="coerce")),
                    _format_report_dt(pd.to_datetime(row.get("end_time"), errors="coerce")),
                    _format_duration(row.get("duration_sec")),
                ])
            return rows

        def _prepare_failure_rows(df: pd.DataFrame) -> List[List[str]]:
            if df.empty:
                return [["No system failure events in this window.", "", "", ""]]
            rows: List[List[str]] = []
            for _, row in df.sort_values("timestamp", ascending=False).head(8).iterrows():
                rows.append([
                    str(row.get("event_type", "")),
                    str(row.get("tag", "")),
                    _format_report_dt(pd.to_datetime(row.get("timestamp"), errors="coerce")),
                    _format_duration(row.get("duration_sec")),
                ])
            return rows

        exceed_rows = _prepare_exceed_rows(exceed_range)
        failure_rows = _prepare_failure_rows(system_events)

        table_ax = ax_tables.inset_axes([0.0, 0.05, 0.48, 0.9])
        table_ax.axis("off")
        table_ax.set_title("Recent Exceedances", fontsize=11, color=R["text"], pad=6)
        exceed_table = table_ax.table(
            cellText=exceed_rows, colLabels=["Tag", "Start", "End", "Duration"],
            cellLoc="left", colLoc="left", loc="center",
        )
        exceed_table.auto_set_font_size(False)
        exceed_table.set_fontsize(9)
        exceed_table.scale(1, 1.2)

        failure_ax = ax_tables.inset_axes([0.52, 0.05, 0.48, 0.9])
        failure_ax.axis("off")
        failure_ax.set_title("System Failures / Errors", fontsize=11, color=R["text"], pad=6)
        failure_table = failure_ax.table(
            cellText=failure_rows, colLabels=["Type", "Tag", "Timestamp", "Duration"],
            cellLoc="left", colLoc="left", loc="center",
        )
        failure_table.auto_set_font_size(False)
        failure_table.set_fontsize(9)
        failure_table.scale(1, 1.2)

        for tbl in (exceed_table, failure_table):
            for (row, _c), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight="bold", color=R["text"])
                    cell.set_facecolor(R["header"])
                else:
                    cell.set_facecolor(R["alt_row"])
                    cell.set_edgecolor(R["border"])

        pdf.savefig(fig)
        plt.close(fig)

    return report_path


def build_report_payload(range_key: str):
    _write_export_lock()
    try:
        report_path = generate_report_pdf(range_key)
    finally:
        _clear_export_lock()
    if not report_path or not os.path.exists(report_path):
        return None
    filename = os.path.basename(report_path)
    return dcc.send_file(report_path, filename=filename)


def build_incident_report_payload(range_key: str):
    _write_export_lock()
    try:
        report_path = generate_incident_report_pdf(range_key)
    finally:
        _clear_export_lock()
    if not report_path or not os.path.exists(report_path):
        return None
    filename = os.path.basename(report_path)
    return dcc.send_file(report_path, filename=filename)


# ============================================================================
#  DATA LOADING -- CSV / JSON readers for raw, hourly, rolling, events, etc.
# ============================================================================


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
        timestamp,date,time,tag,alias,value,status
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


def load_raw_history(max_days: int = 30) -> pd.DataFrame:
    """Load up to `max_days` of raw_data_YYYY-MM-DD.csv files."""

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

    if not os.path.isdir(LOG_DIR):
        return pd.DataFrame(columns=base_cols)

    today = datetime.now().date()
    cutoff_date = today - timedelta(days=max_days - 1)
    # Map file_date -> chosen path. CIP.py compresses raw_data_*.csv to
    # raw_data_*.csv.gz after 7 days, so anything older than a week is gzip.
    # We MUST read both or all-time/history features only see the last 7 days.
    # If both forms exist for the same date (brief window during compression),
    # prefer the uncompressed .csv as the more recent/authoritative copy.
    by_date: Dict[datetime, str] = {}
    for name in os.listdir(LOG_DIR):
        if not name.startswith("raw_data_"):
            continue
        if name.endswith(".csv.gz"):
            date_part = name[len("raw_data_"):-len(".csv.gz")]
            is_plain = False
        elif name.endswith(".csv"):
            date_part = name[len("raw_data_"):-len(".csv")]
            is_plain = True
        else:
            continue
        try:
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except Exception:
            continue
        if file_date < cutoff_date:
            continue
        key = datetime.combine(file_date, datetime.min.time())
        path = os.path.join(LOG_DIR, name)
        # Prefer the plain .csv if we already saw (or later see) one for this date.
        if key not in by_date or is_plain:
            if key in by_date and not is_plain:
                continue
            by_date[key] = path

    if not by_date:
        return pd.DataFrame(columns=base_cols)

    candidates: List[Tuple[datetime, str]] = sorted(by_date.items(), key=lambda tup: tup[0])

    frames: List[pd.DataFrame] = []
    for _, path in candidates:
        try:
            df = pd.read_csv(path)
            for col in base_cols:
                if col not in df.columns:
                    df[col] = None
            frames.append(df[base_cols])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=base_cols)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value_num"] = pd.to_numeric(df.get("value", None), errors="coerce")
    df["status_str"] = df.get("status", "").astype(str)
    df["qa_flag"] = df.get("qa_flag", "OK").fillna("OK")
    df["ok"] = df["qa_flag"].astype(str).str.upper().eq("OK")
    return df


def load_hourly_stats() -> pd.DataFrame:
    """
    Load hourly averages as DataFrame with columns:
        hour_start, hour_end, tag, alias, avg_value, avg_lb_hr, sample_count
    Returns empty DataFrame on any error.
    """
    if not os.path.exists(HOURLY_CSV):
        return pd.DataFrame(
            columns=[
                "hour_start",
                "hour_end",
                "tag",
                "alias",
                "avg_value",
                "avg_lb_hr",
                "sample_count",
            ]
        )
    try:
        df = pd.read_csv(HOURLY_CSV)
    except Exception:
        return pd.DataFrame(
            columns=[
                "hour_start",
                "hour_end",
                "tag",
                "alias",
                "avg_value",
                "avg_lb_hr",
                "sample_count",
            ]
        )

    if "alias" not in df.columns:
        df["alias"] = None
    for c in ["hour_start", "hour_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["avg_value", "avg_lb_hr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_rolling_12hr_stats() -> pd.DataFrame:
    """
    Load rolling 12-hour averages as DataFrame with columns:
        window_start, window_end, tag, alias, avg_value, avg_lb_hr, hours_count
    Returns empty DataFrame on any error.
    """
    if not os.path.exists(ROLLING_12HR_CSV):
        return pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "tag",
                "alias",
                "avg_value",
                "avg_lb_hr",
                "hours_count",
            ]
        )
    try:
        df = pd.read_csv(ROLLING_12HR_CSV)
    except Exception:
        return pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "tag",
                "alias",
                "avg_value",
                "avg_lb_hr",
                "hours_count",
            ]
        )

    if "alias" not in df.columns:
        df["alias"] = None
    for c in ["window_start", "window_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["avg_value", "avg_lb_hr", "hours_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_live_rolling_12hr_stats() -> pd.DataFrame:
    """Load the live rolling 12-hour snapshot (CIP.py overwrites this each poll).

    Schema is the same as the historical rolling CSV plus a `generated_at` column.
    Returns an empty DataFrame if the file does not exist or cannot be parsed.
    """
    columns = [
        "window_start",
        "window_end",
        "tag",
        "alias",
        "avg_value",
        "avg_lb_hr",
        "hours_count",
        "generated_at",
    ]
    if not os.path.exists(LIVE_ROLLING_12HR_CSV):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(LIVE_ROLLING_12HR_CSV)
    except Exception:
        return pd.DataFrame(columns=columns)

    if "alias" not in df.columns:
        df["alias"] = None
    for c in ["window_start", "window_end", "generated_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["avg_value", "avg_lb_hr", "hours_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_live_rolling_12hr(
    rolling_df: pd.DataFrame, live_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine the historical rolling 12-hr CSV with the live snapshot.

    The live snapshot represents the in-progress 12-hr window (current partial
    hour included). Where it overlaps the historical record we keep the live
    row so the dashboard reflects the value being written to the PLC right now.
    """
    if live_df is None or live_df.empty:
        return rolling_df
    historical_cols = [
        "window_start",
        "window_end",
        "tag",
        "alias",
        "avg_value",
        "avg_lb_hr",
        "hours_count",
    ]
    live_subset = live_df[[c for c in historical_cols if c in live_df.columns]].copy()
    if rolling_df is None or rolling_df.empty:
        return live_subset

    combined = pd.concat([rolling_df, live_subset], ignore_index=True, sort=False)
    if "window_end" in combined.columns:
        combined = combined.sort_values("window_end")
    return combined


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


def load_exceedances() -> pd.DataFrame:
    if not os.path.exists(EXCEEDANCES_CSV):
        return pd.DataFrame(
            columns=[
                "tag",
                "start_time",
                "end_time",
                "duration_sec",
                "max_value",
                "avg_value_over_event",
                "limit_value",
            ]
        )
    try:
        df = pd.read_csv(EXCEEDANCES_CSV)
    except Exception:
        return pd.DataFrame(
            columns=[
                "tag",
                "start_time",
                "end_time",
                "duration_sec",
                "max_value",
                "avg_value_over_event",
                "limit_value",
            ]
        )

    for col in ["start_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT
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


def load_system_health() -> Dict[str, object]:
    """Load system_health.json if available."""
    default = {
        "status": "Unknown",
        "status_reason": "No health file yet",
        "last_poll_success_ts": None,
        "last_poll_error_ts": None,
        "error_count_last_hour": None,
        "disk_free_GB": None,
        "log_dir_size_GB": None,
    }
    if not os.path.exists(SYSTEM_HEALTH_JSON):
        return default
    try:
        with open(SYSTEM_HEALTH_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            default.update(data)
    except Exception:
        default["status"] = "Error"
        default["status_reason"] = "Failed to read system_health.json"
    return default


def _overlap_seconds(start: datetime, end: datetime, window_start: datetime, window_end: datetime) -> float:
    latest_start = max(start, window_start)
    earliest_end = min(end, window_end)
    delta = (earliest_end - latest_start).total_seconds()
    return max(0.0, delta)


def detect_exceedance_events(
    raw_df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, object]],
    merge_gap: int = 1,
    stable_samples: int = 2,
) -> pd.DataFrame:
    """
    Scan raw data for limit exceedances and aggregate them into events.

    Args:
        raw_df: Raw samples across days.
        thresholds: Per-tag thresholds containing high_limit/low_limit.
        merge_gap: Number of consecutive invalid/missing samples to tolerate inside an event.
        stable_samples: Consecutive in-range samples required to close an event.
    """

    columns = [
        "tag",
        "start_time",
        "end_time",
        "duration_sec",
        "max_value",
        "avg_value_over_event",
        "limit_value",
    ]

    if raw_df.empty or "tag" not in raw_df.columns:
        return pd.DataFrame(columns=columns)

    df = raw_df.copy()
    df = df.sort_values("timestamp")
    df["value_num"] = pd.to_numeric(df.get("value_num", df.get("value", None)), errors="coerce")
    df["ok"] = df.get("ok", True)

    events: List[Dict[str, object]] = []

    for tag, group in df.groupby("tag"):
        entry = thresholds.get(tag, {}) if isinstance(thresholds.get(tag, {}), dict) else {}
        low_limit = _to_float(entry.get("low_limit"))
        high_limit = _to_float(entry.get("high_limit"))

        if low_limit is None and high_limit is None:
            continue

        active = False
        start_ts: Optional[datetime] = None
        last_exceed_ts: Optional[datetime] = None
        values: List[float] = []
        limit_value: Optional[float] = None
        stable_count = 0
        gap_run = 0

        for _, row in group.iterrows():
            ts = row.get("timestamp")
            if pd.isna(ts):
                continue
            ts = pd.to_datetime(ts)
            ok = bool(row.get("ok", False))
            val = row.get("value_num")
            try:
                num_val = float(val)
            except Exception:
                num_val = float("nan")

            if not ok or num_val != num_val:
                if active:
                    gap_run += 1
                    stable_count = 0
                    if gap_run > merge_gap and start_ts and last_exceed_ts:
                        duration = max(1, int((last_exceed_ts - start_ts).total_seconds()))
                        events.append(
                            {
                                "tag": str(tag),
                                "start_time": start_ts,
                                "end_time": last_exceed_ts,
                                "duration_sec": duration,
                                "max_value": max(values) if values else float("nan"),
                                "avg_value_over_event": float(sum(values) / len(values)) if values else float("nan"),
                                "limit_value": limit_value,
                            }
                        )
                        active = False
                        values = []
                        limit_value = None
                continue

            gap_run = 0

            exceeds_high = high_limit is not None and num_val > high_limit
            exceeds_low = low_limit is not None and num_val < low_limit
            exceeds = exceeds_high or exceeds_low

            if exceeds:
                if not active:
                    active = True
                    start_ts = ts
                    values = []
                    limit_value = high_limit if exceeds_high else low_limit
                last_exceed_ts = ts
                values.append(num_val)
                stable_count = 0
                continue

            # In-range sample
            if active and start_ts and last_exceed_ts:
                stable_count += 1
                if stable_count >= stable_samples:
                    duration = max(1, int((last_exceed_ts - start_ts).total_seconds()))
                    events.append(
                        {
                            "tag": str(tag),
                            "start_time": start_ts,
                            "end_time": last_exceed_ts,
                            "duration_sec": duration,
                            "max_value": max(values) if values else float("nan"),
                            "avg_value_over_event": float(sum(values) / len(values)) if values else float("nan"),
                            "limit_value": limit_value,
                        }
                    )
                    active = False
                    values = []
                    limit_value = None
                    stable_count = 0

        if active and start_ts and last_exceed_ts:
            duration = max(1, int((last_exceed_ts - start_ts).total_seconds()))
            events.append(
                {
                    "tag": str(tag),
                    "start_time": start_ts,
                    "end_time": last_exceed_ts,
                    "duration_sec": duration,
                    "max_value": max(values) if values else float("nan"),
                    "avg_value_over_event": float(sum(values) / len(values)) if values else float("nan"),
                    "limit_value": limit_value,
                }
            )

    if not events:
        return pd.DataFrame(columns=columns)

    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values("start_time", ascending=False)
    return events_df


def save_exceedances(events_df: pd.DataFrame) -> None:
    ensure_dir(os.path.dirname(EXCEEDANCES_CSV) or ".")
    headers = [
        "tag",
        "start_time",
        "end_time",
        "duration_sec",
        "max_value",
        "avg_value_over_event",
        "limit_value",
    ]
    if events_df.empty:
        ensure_csv(EXCEEDANCES_CSV, headers)
        return

    try:
        df = events_df.copy()
        df["start_time"] = df["start_time"].astype(str)
        df["end_time"] = df["end_time"].astype(str)
        df.to_csv(EXCEEDANCES_CSV, index=False, columns=headers)
    except Exception:
        pass


def save_thresholds(th: Dict[str, Dict[str, float]]) -> None:
    ensure_dir(os.path.dirname(THRESHOLDS_JSON) or ".")
    try:
        with open(THRESHOLDS_JSON, "w", encoding="utf-8") as f:
            json.dump(th, f, indent=2)
    except Exception:
        pass


# ============================================================================
#  SETTINGS & TAG UTILITIES -- alias maps, tag parsing, EPA settings
# ============================================================================


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


def load_alias_map_from_settings() -> Dict[str, str]:
    """
    Return {tag: alias} from settings.json (if available).
    """
    if not os.path.exists(SETTINGS_JSON):
        return {}
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    tags_text = data.get("tags", "")
    _, alias_map = parse_tags_and_aliases(tags_text)
    return alias_map


def load_machine_state_tag_from_settings() -> str:
    if not os.path.exists(SETTINGS_JSON):
        return ""
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("machine_state_tag", "") or "").strip()


def load_feed_settings() -> Dict[str, str]:
    """Load conveyor / feed system detection settings from settings.json.

    Returns dict with keys:
        detection  -- "none", "weight", or "tag"
        weight_tag -- PLC tag for the hopper/silo weight
        weight_units -- display units for the weight value
        conveyor_tag -- PLC tag for direct conveyor on/off signal
    """
    default = {"detection": "none", "weight_tag": "", "weight_units": "", "conveyor_tag": ""}
    if not os.path.exists(SETTINGS_JSON):
        return default
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    if not isinstance(data, dict):
        return default

    prod = data.get("production_tracking", {})
    if not isinstance(prod, dict):
        return default

    tags_text = data.get("tags", "")
    _, alias_map = parse_tags_and_aliases(tags_text)

    detection = str(prod.get("conveyor_detection", "none") or "none").strip().lower()
    if detection not in ("none", "weight", "tag"):
        detection = "none"

    weight_tag = resolve_tag_from_alias(str(prod.get("weight_tag", "") or ""), alias_map)
    conveyor_tag = resolve_tag_from_alias(str(prod.get("conveyor_tag", "") or ""), alias_map)
    weight_units = str(prod.get("weight_units", "") or "")

    return {
        "detection": detection,
        "weight_tag": weight_tag,
        "weight_units": weight_units,
        "conveyor_tag": conveyor_tag,
    }


def compute_feed_status(
    raw_df: pd.DataFrame,
    feed_settings: Dict[str, str],
) -> Dict[str, object]:
    """Determine whether the feed system / conveyor is currently running.

    Weight-based: looks at the last ~5 minutes of weight samples. If the
    weight shows a steady decrease (linear slope < 0) the conveyor is
    considered ON. Large positive spikes (refills) are filtered out so
    they don't mask a real feeding trend.

    Tag-based: simply checks the latest value of the conveyor tag.

    Returns dict:
        running      -- bool or None (unknown)
        method       -- "weight" | "tag" | "none"
        feed_rate    -- float (units/min, weight-based only) or None
        weight_units -- str
        detail       -- short human-readable explanation
    """
    detection = feed_settings.get("detection", "none")
    result: Dict[str, object] = {
        "running": None,
        "method": detection,
        "feed_rate": None,
        "weight_units": feed_settings.get("weight_units", ""),
        "detail": "",
    }

    if detection == "none" or raw_df is None or raw_df.empty:
        result["detail"] = "Feed detection not configured"
        return result

    if "tag" not in raw_df.columns or "timestamp" not in raw_df.columns:
        result["detail"] = "No data available"
        return result

    # --- Tag-based detection ---
    if detection == "tag":
        tag = feed_settings.get("conveyor_tag", "")
        if not tag:
            result["detail"] = "Conveyor tag not set"
            return result
        tag_df = raw_df.loc[raw_df["tag"] == tag].copy()
        if tag_df.empty:
            result["detail"] = f"No samples for {tag}"
            return result
        tag_df = tag_df.sort_values("timestamp")
        latest_val = tag_df.iloc[-1].get("value")
        try:
            num = float(latest_val)
            running = num > 0
        except (TypeError, ValueError):
            running = str(latest_val).strip().lower() in ("true", "1", "on", "yes")
        result["running"] = running
        result["detail"] = "ON" if running else "OFF"
        return result

    # --- Weight-based detection ---
    if detection == "weight":
        tag = feed_settings.get("weight_tag", "")
        if not tag:
            result["detail"] = "Weight tag not set"
            return result
        tag_df = raw_df.loc[raw_df["tag"] == tag].copy()
        if tag_df.empty:
            result["detail"] = f"No samples for {tag}"
            return result

        tag_df = tag_df.sort_values("timestamp").reset_index(drop=True)
        tag_df["_val"] = pd.to_numeric(tag_df.get("value", None), errors="coerce")
        tag_df = tag_df.dropna(subset=["_val", "timestamp"])
        if len(tag_df) < 3:
            result["detail"] = "Not enough samples"
            return result

        # Use last 5 minutes of data
        cutoff = tag_df["timestamp"].max() - timedelta(minutes=5)
        window = tag_df.loc[tag_df["timestamp"] >= cutoff].copy()
        if len(window) < 3:
            window = tag_df.tail(10).copy()

        # Filter out large positive spikes (refills): any sample-to-sample
        # increase > 20% of the window range is treated as a refill and the
        # sample after the jump is dropped.
        vals = window["_val"].values
        diffs = pd.Series(vals).diff().fillna(0).values
        val_range = float(max(vals) - min(vals)) if len(vals) > 1 else 1.0
        spike_threshold = max(val_range * 0.20, 1.0)
        keep_mask = diffs <= spike_threshold
        keep_mask[0] = True  # always keep first
        window = window.loc[window.index[keep_mask]]

        if len(window) < 3:
            result["detail"] = "Insufficient data after spike filtering"
            return result

        # Compute linear slope (units per second)
        t_seconds = (window["timestamp"] - window["timestamp"].iloc[0]).dt.total_seconds().values
        w_vals = window["_val"].values
        if t_seconds[-1] - t_seconds[0] < 1:
            result["detail"] = "Time span too short"
            return result

        n = len(t_seconds)
        sum_t = t_seconds.sum()
        sum_w = w_vals.sum()
        sum_tw = (t_seconds * w_vals).sum()
        sum_tt = (t_seconds * t_seconds).sum()
        denom = n * sum_tt - sum_t * sum_t
        if abs(denom) < 1e-12:
            result["detail"] = "Cannot compute slope"
            return result

        slope_per_sec = (n * sum_tw - sum_t * sum_w) / denom  # units/sec

        # Negative slope means weight is decreasing = conveyor feeding
        units = feed_settings.get("weight_units", "") or "units"
        if slope_per_sec < -0.001:
            feed_rate_per_min = abs(slope_per_sec) * 60.0
            result["running"] = True
            result["feed_rate"] = round(feed_rate_per_min, 2)
            result["detail"] = f"Feeding at {feed_rate_per_min:.1f} {units}/min"
        else:
            result["running"] = False
            result["feed_rate"] = 0.0
            result["detail"] = "Weight stable or increasing"

        return result

    return result


def compute_weight_decrease_time_range(
    raw_df: pd.DataFrame,
    weight_tag: str,
    start: datetime,
    end: datetime,
    max_gap_sec: int = 120,
    window_sec: int = 300,
    refill_spike_pct: float = 0.20,
) -> Dict[str, object]:
    """Compute conveyor run-time from weight-decrease patterns over a range.

    This always runs when a weight tag is configured, even if the primary
    conveyor detection method is tag-based or disabled. The result is a
    secondary / corroborating metric useful for reports.

    Algorithm:
    - Walk weight samples in sliding windows of `window_sec` seconds.
    - For each window, fit a linear slope.
    - Filter out refill spikes (large positive jumps) before fitting.
    - If the slope is negative (weight decreasing), attribute the window's
      elapsed time to "conveyor running".
    - Cap individual inter-sample gaps at `max_gap_sec` to avoid inflating
      totals from polling outages.

    Args:
        raw_df: Raw samples (possibly multi-day).
        weight_tag: Tag that reports hopper / silo weight.
        start, end: Window boundaries (inclusive start, exclusive end).
        max_gap_sec: Cap per inter-sample gap (default 120 s).
        window_sec: Sliding-window size for slope detection (default 300 s).
        refill_spike_pct: Sample-to-sample increase fraction of the window
            range that counts as a refill and is filtered out.

    Returns dict:
        total_minutes          -- minutes where weight was decreasing
        total_hours            -- total_minutes / 60
        avg_feed_rate_per_min  -- average consumption rate (units/min) when feeding, or None
        weight_units           -- units label (passthrough for display)
        sample_count           -- number of OK weight samples examined
        capped_gap_count       -- polling gaps that exceeded the cap
    """
    result: Dict[str, object] = {
        "total_minutes": 0.0,
        "total_hours": 0.0,
        "avg_feed_rate_per_min": None,
        "weight_units": "",
        "sample_count": 0,
        "capped_gap_count": 0,
    }

    if (
        not weight_tag
        or raw_df is None
        or raw_df.empty
        or "tag" not in raw_df.columns
        or "timestamp" not in raw_df.columns
    ):
        return result

    w_df = raw_df.loc[raw_df["tag"] == weight_tag].copy()
    if w_df.empty:
        return result

    w_df["timestamp"] = pd.to_datetime(w_df["timestamp"], errors="coerce")
    w_df = w_df.dropna(subset=["timestamp"])
    w_df = w_df.loc[(w_df["timestamp"] >= start) & (w_df["timestamp"] < end)]
    if w_df.empty:
        return result

    if "qa_flag" in w_df.columns:
        w_df = w_df.loc[w_df["qa_flag"].astype(str).str.upper().eq("OK")]
    if w_df.empty:
        return result

    w_df = w_df.sort_values("timestamp").reset_index(drop=True)
    w_df["_val"] = pd.to_numeric(w_df.get("value", None), errors="coerce")
    w_df = w_df.dropna(subset=["_val"])
    if len(w_df) < 3:
        return result

    # Raw inter-sample gaps (seconds); clamp negatives from clock jumps to 0
    raw_diffs = w_df["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0)
    capped_gap_count = int((raw_diffs > max_gap_sec).sum())
    w_df["_dt"] = raw_diffs.clip(upper=max_gap_sec)

    # Filter refill spikes: drop samples where value jumped up by more than
    # refill_spike_pct of the overall value range
    vals = w_df["_val"].values
    val_range = float(vals.max() - vals.min()) if len(vals) > 1 else 1.0
    spike_threshold = max(val_range * refill_spike_pct, 1.0)
    val_diffs = pd.Series(vals).diff().fillna(0).values
    keep_mask = val_diffs <= spike_threshold
    keep_mask[0] = True
    w_df = w_df.loc[w_df.index[keep_mask]].reset_index(drop=True)
    if len(w_df) < 3:
        return result

    # Sliding-window slope detection
    timestamps = w_df["timestamp"]
    values = w_df["_val"].values
    t_sec = (timestamps - timestamps.iloc[0]).dt.total_seconds().values

    total_decrease_seconds = 0.0
    total_weight_consumed = 0.0

    # For each sample, look back up to window_sec and fit slope
    # Attribute the sample's preceding gap as "conveyor running" if slope < 0
    for i in range(1, len(w_df)):
        # Find window start index
        t_end = t_sec[i]
        window_start_time = t_end - window_sec
        j = i
        while j > 0 and t_sec[j - 1] >= window_start_time:
            j -= 1
        if i - j < 2:
            continue

        wt = t_sec[j:i + 1]
        wv = values[j:i + 1]
        n = len(wt)
        sum_t = wt.sum()
        sum_v = wv.sum()
        sum_tv = (wt * wv).sum()
        sum_tt = (wt * wt).sum()
        denom = n * sum_tt - sum_t * sum_t
        if abs(denom) < 1e-12:
            continue
        slope_per_sec = (n * sum_tv - sum_t * sum_v) / denom

        if slope_per_sec < -0.001:
            gap_sec = float(w_df["_dt"].iloc[i])
            total_decrease_seconds += gap_sec
            total_weight_consumed += abs(slope_per_sec) * gap_sec

    total_minutes = total_decrease_seconds / 60.0
    avg_rate = None
    if total_decrease_seconds > 0:
        avg_rate = round((total_weight_consumed / total_decrease_seconds) * 60.0, 2)

    result["total_minutes"] = round(total_minutes, 2)
    result["total_hours"] = round(total_minutes / 60.0, 2)
    result["avg_feed_rate_per_min"] = avg_rate
    result["sample_count"] = int(len(w_df))
    result["capped_gap_count"] = capped_gap_count
    return result


def compute_processing_time_minutes(
    raw_df: pd.DataFrame,
    machine_state_tag: str,
) -> Tuple[float, float]:
    """
    Compute processing time (minutes) for the current hour from raw data.

    Looks at the machine state tag in raw samples. State == 3 means Processing.
    Estimates time in Processing by counting consecutive sample intervals where
    the machine was in state 3.

    Returns:
        (current_hour_processing_minutes, today_total_processing_minutes)
    """
    if (
        not machine_state_tag
        or raw_df is None
        or raw_df.empty
        or "tag" not in raw_df.columns
    ):
        return float("nan"), float("nan")

    state_df = raw_df.loc[raw_df["tag"] == machine_state_tag].copy()
    if state_df.empty:
        return float("nan"), float("nan")

    if "timestamp" not in state_df.columns:
        return float("nan"), float("nan")

    # Honor qa_flag -- only OK samples contribute, matching the precise
    # report-side calculation (compute_processing_time_range) so the on-screen
    # "today / current hour" figure agrees with the PDF reports.
    if "qa_flag" in state_df.columns:
        state_df = state_df.loc[
            state_df["qa_flag"].astype(str).str.upper().eq("OK")
        ]
    if state_df.empty:
        return float("nan"), float("nan")

    state_df = state_df.sort_values("timestamp").reset_index(drop=True)
    state_df["_val"] = pd.to_numeric(state_df.get("value", None), errors="coerce")

    # Determine processing state: value rounds to 3
    state_df["_is_processing"] = (state_df["_val"] >= 2.5) & (state_df["_val"] <= 3.5)

    # Calculate interval between consecutive samples
    state_df["_dt"] = state_df["timestamp"].diff().dt.total_seconds().fillna(0)
    # Negative gaps (clock step-backs / duplicate timestamps) -> 0, then cap
    # individual intervals at 120s to avoid inflating from gaps. Mirrors
    # compute_processing_time_range so the two calculations stay consistent.
    state_df["_dt"] = state_df["_dt"].clip(lower=0, upper=120)

    # Current hour
    now = datetime.now()
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    hour_end = hour_start + timedelta(hours=1)
    current_mask = (
        (state_df["timestamp"] >= hour_start)
        & (state_df["timestamp"] < hour_end)
        & state_df["_is_processing"]
    )
    current_hour_seconds = state_df.loc[current_mask, "_dt"].sum()
    current_hour_minutes = current_hour_seconds / 60.0

    # Today total
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_mask = (
        (state_df["timestamp"] >= today_start)
        & state_df["_is_processing"]
    )
    today_seconds = state_df.loc[today_mask, "_dt"].sum()
    today_total_minutes = today_seconds / 60.0

    return current_hour_minutes, today_total_minutes


# Cache for all-time processing total. Recomputing across every raw_data_*.csv
# is O(GB) of disk I/O; the value changes slowly enough that a 5-minute TTL
# keeps the dashboard responsive without misleading the operator.
_ALL_TIME_PROC_CACHE: Dict[str, object] = {
    "value_minutes": None,
    "computed_at": 0.0,
    "ttl_sec": 300.0,
    "machine_state_tag": None,
}


def compute_all_time_processing_minutes(machine_state_tag: str) -> Optional[float]:
    """Estimated total processing minutes over all history, from hourly_averages.csv.

    Derived from the complete (and small) hourly_averages.csv rather than the
    raw_data_*.csv files. Reading every raw file -- now mostly gzipped and
    spanning months -- on each refresh would stall the dashboard, so the live
    "All time" figure uses the hourly state-tag average instead: every hour
    whose machine-state average rounds to Processing (state 3, i.e. in
    [2.5, 3.5]) contributes 60 minutes.

    This is an estimate. Over- and under-counted partial hours largely cancel,
    so at scale it tracks the precise raw calculation closely (validated within
    ~0.2% on production data). The PDF reports still use the exact sub-minute
    raw calculation (compute_processing_time_range), which now reads gzipped
    history too.

    Cached for ``_ALL_TIME_PROC_CACHE['ttl_sec']`` seconds. Returns None if no
    machine state tag is configured or no hourly history exists. Cache is
    invalidated when the configured machine state tag changes.
    """
    import time as _time
    if not machine_state_tag:
        return None

    now_ts = _time.time()
    cached_tag = _ALL_TIME_PROC_CACHE.get("machine_state_tag")
    cached_val = _ALL_TIME_PROC_CACHE.get("value_minutes")
    age = now_ts - float(_ALL_TIME_PROC_CACHE.get("computed_at", 0.0))
    ttl = float(_ALL_TIME_PROC_CACHE.get("ttl_sec", 300.0))
    if (
        cached_val is not None
        and cached_tag == machine_state_tag
        and age < ttl
    ):
        return float(cached_val)

    # Cache miss / stale -- recompute from the complete hourly file.
    try:
        hourly = load_hourly_stats()
    except Exception:
        return cached_val if cached_val is not None else None
    if hourly is None or hourly.empty or "tag" not in hourly.columns:
        return None

    state_rows = hourly.loc[hourly["tag"] == machine_state_tag]
    if state_rows.empty:
        return None

    avg = pd.to_numeric(state_rows.get("avg_value"), errors="coerce")
    processing_hours = int(((avg >= 2.5) & (avg <= 3.5)).sum())
    total = float(processing_hours * 60.0)

    _ALL_TIME_PROC_CACHE["value_minutes"] = total
    _ALL_TIME_PROC_CACHE["computed_at"] = now_ts
    _ALL_TIME_PROC_CACHE["machine_state_tag"] = machine_state_tag
    return total


def compute_processing_time_range(
    raw_df: pd.DataFrame,
    machine_state_tag: str,
    start: datetime,
    end: datetime,
    max_gap_sec: int = 120,
) -> Dict[str, object]:
    """Compute processing time for an arbitrary range using sub-minute precision.

    This is the canonical, precise calculation used by PDF reports. It walks
    the raw machine-state samples and sums inter-sample gaps where the state
    was 3 (Processing).

    Improvements over hour-average counting:
    - Sub-minute granularity instead of whole-hour counting
    - Honors the qa_flag column (only OK samples contribute)
    - Tracks gaps that exceeded the cap so the caller can warn
    - Returns the set of hours that contained ANY processing time, not just
      hours whose state-tag average rounded to 3

    Args:
        raw_df: Raw samples (possibly multi-day).
        machine_state_tag: Tag that reports machine state (3 = Processing).
        start, end: Window boundaries (inclusive start, exclusive end).
        max_gap_sec: Cap on each inter-sample gap. Prevents long polling
            outages from distorting the total. Default 120 s.

    Returns dict:
        total_minutes           -- precise processing minutes in range
        total_hours             -- total_minutes / 60
        hour_starts_with_any    -- set of pandas.Timestamp hour-floored values
                                   (for weight filtering; comparable directly
                                   against the parsed hour_start column)
        capped_gap_count        -- number of gaps that hit the cap (data-gap indicator)
        sample_count            -- number of OK state samples examined
    """
    result: Dict[str, object] = {
        "total_minutes": 0.0,
        "total_hours": 0.0,
        "hour_starts_with_any": set(),
        "capped_gap_count": 0,
        "sample_count": 0,
    }

    if (
        not machine_state_tag
        or raw_df is None
        or raw_df.empty
        or "tag" not in raw_df.columns
        or "timestamp" not in raw_df.columns
    ):
        return result

    state_df = raw_df.loc[raw_df["tag"] == machine_state_tag].copy()
    if state_df.empty:
        return result

    # Ensure timestamps are datetimes and filter to the requested range
    state_df["timestamp"] = pd.to_datetime(state_df["timestamp"], errors="coerce")
    state_df = state_df.dropna(subset=["timestamp"])
    state_df = state_df.loc[
        (state_df["timestamp"] >= start) & (state_df["timestamp"] < end)
    ]
    if state_df.empty:
        return result

    # Honor qa_flag — only "OK" samples contribute (filters out PLC_ERROR, STALE, etc.)
    if "qa_flag" in state_df.columns:
        state_df = state_df.loc[
            state_df["qa_flag"].astype(str).str.upper().eq("OK")
        ]
    if state_df.empty:
        return result

    state_df = state_df.sort_values("timestamp").reset_index(drop=True)
    state_df["_val"] = pd.to_numeric(state_df.get("value", None), errors="coerce")
    state_df = state_df.dropna(subset=["_val"])
    if state_df.empty:
        return result

    state_df["_is_processing"] = (state_df["_val"] >= 2.5) & (state_df["_val"] <= 3.5)

    # Raw inter-sample gaps (seconds)
    raw_diffs = state_df["timestamp"].diff().dt.total_seconds().fillna(0)
    # Negative gaps (clock jumps) -> treat as 0
    raw_diffs = raw_diffs.clip(lower=0)
    # Count gaps that would exceed the cap BEFORE clipping so caller can warn
    capped_gap_count = int((raw_diffs > max_gap_sec).sum())
    # Cap to avoid inflating totals
    state_df["_dt"] = raw_diffs.clip(upper=max_gap_sec)

    # Processing gaps only
    proc_mask = state_df["_is_processing"]
    processing_seconds = float(state_df.loc[proc_mask, "_dt"].sum())
    total_minutes = processing_seconds / 60.0

    # Which hours contained any processing time? Use the LATER sample's hour
    # (i.e. the hour the gap ended in — consistent with how gaps are attributed).
    # Stored as pandas.Timestamp objects so consumers can compare directly
    # against the parsed-datetime hour_start columns in load_hourly_stats()
    # without any string-format brittleness.
    hour_starts_with_any: set = set()
    if proc_mask.any():
        proc_rows = state_df.loc[proc_mask & (state_df["_dt"] > 0)]
        if not proc_rows.empty:
            hour_starts = proc_rows["timestamp"].dt.floor("h")
            hour_starts_with_any = {hs for hs in hour_starts.unique() if pd.notna(hs)}

    result["total_minutes"] = round(total_minutes, 2)
    result["total_hours"] = round(total_minutes / 60.0, 2)
    result["hour_starts_with_any"] = hour_starts_with_any
    result["capped_gap_count"] = capped_gap_count
    result["sample_count"] = int(len(state_df))
    return result


def compute_cems_uptime(
    hourly_df: pd.DataFrame,
    machine_state_tag: str,
    cems_tags: List[str],
    start: datetime,
    end: datetime,
) -> Dict[str, object]:
    """Calculate CEMS data availability against operating hours.

    For each hour in [start, end) where the machine was in state 3
    (Processing), check whether at least one CEMS tag has valid data
    (sample_count > 0).  Returns stats for the period.

    Args:
        hourly_df:  Full hourly_averages dataframe.
        machine_state_tag:  Tag that reports machine state (3 = Processing).
        cems_tags:  List of CEMS tag names (CO, NOx, O2) to check.
        start, end: Period boundaries.

    Returns dict:
        operating_hours   -- int, hours machine was processing
        valid_cems_hours  -- int, operating hours with CEMS data
        missing_hours     -- int, operating hours WITHOUT CEMS data
        uptime_pct        -- float, valid / operating * 100  (0 if no op hrs)
        total_hours       -- int, calendar hours in the period
        period_label      -- str, human-readable period description
    """
    result: Dict[str, object] = {
        "operating_hours": 0,
        "valid_cems_hours": 0,
        "missing_hours": 0,
        "uptime_pct": 0.0,
        "total_hours": 0,
        "period_label": "",
    }

    if hourly_df is None or hourly_df.empty or not machine_state_tag:
        return result

    if "hour_start" not in hourly_df.columns or "tag" not in hourly_df.columns:
        return result

    # Filter to the requested time window
    df = hourly_df.copy()
    df["hour_start"] = pd.to_datetime(df["hour_start"], errors="coerce")
    windowed = df.loc[(df["hour_start"] >= start) & (df["hour_start"] < end)]
    if windowed.empty:
        return result

    # Total calendar hours in the period
    total_hours = max(1, int((end - start).total_seconds() / 3600))
    result["total_hours"] = total_hours

    # Find operating hours (machine state ~= 3)
    state_rows = windowed.loc[windowed["tag"] == machine_state_tag].copy()
    state_vals = pd.to_numeric(state_rows.get("avg_value"), errors="coerce")
    state_rows = state_rows.assign(_state=state_vals)
    processing = state_rows.loc[
        (state_rows["_state"] >= 2.5) & (state_rows["_state"] <= 3.5)
    ]
    operating_hour_starts = set(processing["hour_start"])
    operating_hours = len(operating_hour_starts)
    result["operating_hours"] = operating_hours

    if operating_hours == 0:
        return result

    # For each operating hour check if any CEMS tag has sample_count > 0
    cems_tags_set = set(t for t in cems_tags if t)
    if not cems_tags_set:
        # No CEMS tags configured — count any non-state tag with data
        cems_data = windowed.loc[
            (windowed["tag"] != machine_state_tag)
            & (windowed["hour_start"].isin(operating_hour_starts))
        ]
    else:
        cems_data = windowed.loc[
            (windowed["tag"].isin(cems_tags_set))
            & (windowed["hour_start"].isin(operating_hour_starts))
        ]

    # An hour counts as "valid" if at least one CEMS tag has sample_count > 0
    if not cems_data.empty:
        sample_counts = pd.to_numeric(cems_data.get("sample_count"), errors="coerce").fillna(0)
        cems_data = cems_data.assign(_cnt=sample_counts)
        valid_hours_set = set(
            cems_data.loc[cems_data["_cnt"] > 0, "hour_start"]
        )
    else:
        valid_hours_set = set()

    valid_cems_hours = len(valid_hours_set & operating_hour_starts)
    missing_hours = operating_hours - valid_cems_hours

    result["valid_cems_hours"] = valid_cems_hours
    result["missing_hours"] = missing_hours
    result["uptime_pct"] = round(
        (valid_cems_hours / operating_hours) * 100.0, 1
    ) if operating_hours > 0 else 0.0

    return result


def _merge_alias_from_df(df: pd.DataFrame, alias_map: Dict[str, str]) -> None:
    if df is None or df.empty:
        return
    if "tag" not in df.columns or "alias" not in df.columns:
        return
    try:
        pairs = df[["tag", "alias"]].dropna()
    except Exception:
        return
    for tag, alias in pairs.itertuples(index=False):
        tag_str = str(tag).strip()
        alias_str = str(alias).strip()
        if tag_str and alias_str:
            alias_map.setdefault(tag_str, alias_str)


def build_alias_lookup(
    raw_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    settings_alias_map: Dict[str, str],
) -> Dict[str, str]:
    alias_map = dict(settings_alias_map)
    _merge_alias_from_df(raw_df, alias_map)
    _merge_alias_from_df(hourly_df, alias_map)
    _merge_alias_from_df(rolling_df, alias_map)
    return alias_map


def parse_tags_and_aliases(text: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Parse tags from settings text and return (tags, alias_map).
    Supports optional aliases in the format "Tag|Alias".
    """
    if not isinstance(text, str):
        return [], {}

    tags: List[str] = []
    alias_map: Dict[str, str] = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if "|" in raw:
            tag_part, alias_part = raw.split("|", 1)
            tag = tag_part.strip()
            alias = alias_part.strip()
        else:
            tag = raw
            alias = ""
        if not tag:
            continue
        tags.append(tag)
        if alias:
            alias_map[tag] = alias
    return tags, alias_map


def resolve_tag_from_alias(name: str, alias_map: Dict[str, str]) -> str:
    if not name:
        return ""
    if name in alias_map:
        return name
    normalized = name.strip().casefold()
    for tag, alias in alias_map.items():
        if isinstance(alias, str) and alias.strip().casefold() == normalized:
            return tag
    return name


def load_epa_settings() -> Dict[str, object]:
    if not os.path.exists(SETTINGS_JSON):
        return {}
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    tags_text = data.get("tags", "")
    _, alias_map = parse_tags_and_aliases(tags_text)
    return {
        "epa_enabled": bool(data.get("epa_enabled", False)),
        "epa_flow_tag": resolve_tag_from_alias(
            str(data.get("epa_flow_tag", "") or ""), alias_map
        ),
        "epa_o2_tag": resolve_tag_from_alias(
            str(data.get("epa_o2_tag", "") or ""), alias_map
        ),
        "epa_o2_units": str(data.get("epa_o2_units", "percent") or "percent"),
        "epa_ref_o2_pct": float(data.get("epa_ref_o2_pct", 3.0) or 3.0),
        "epa_nox_tag": resolve_tag_from_alias(
            str(data.get("epa_nox_tag", "") or ""), alias_map
        ),
        "epa_co_tag": resolve_tag_from_alias(
            str(data.get("epa_co_tag", "") or ""), alias_map
        ),
    }


# ============================================================================
#  GAUGE HELPERS -- classification, status colour, gauge-range computation
# ============================================================================


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
        return COLOR_GOOD
    if status == "warning":
        return COLOR_WARNING
    if status == "bad":
        return COLOR_BAD
    return COLOR_TEXT_SUBTLE


def compute_gauge_range(
    low: Optional[float],
    high: Optional[float],
    sample_values: List[float],
    regulatory_high: Optional[float] = None,
    regulatory_low: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute (low, high, gauge_min, gauge_max) for a tag.
    - If thresholds exist, use them (and swap if low/high reversed).
    - Otherwise derive a reasonable range from sample values or fall back to [0, 1].
    - The gauge max is set sensibly above the highest defined threshold so the
      red (exceedance) band is clearly visible — 20% headroom above a permit
      limit, or 25% above an operational limit when no permit limit exists.
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
    elif values:
        # No thresholds: derive from data
        vmin = min(values)
        vmax = max(values)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        low, high = vmin, vmax
        center = (vmin + vmax) / 2.0
        span = max(1.0, (vmax - vmin) * 1.5)
        gmin = round(center - span / 2.0, 2)
        gmax = round(center + span / 2.0, 2)
    else:
        low, high = 0.0, 1.0
        gmin, gmax = 0.0, 1.0

    # Ensure the gauge max sits sensibly above the highest threshold so the
    # red exceedance band has visible space. Prefer 20% above a permit limit,
    # otherwise 25% above an operational high.
    if regulatory_high is not None:
        gmax = max(gmax, round(regulatory_high * 1.2, 2))
    elif high is not None:
        gmax = max(gmax, round(high * 1.25, 2))

    # Likewise on the low side, give the red band room below a low limit.
    if regulatory_low is not None:
        if regulatory_low > 0:
            gmin = min(gmin, round(regulatory_low * 0.8, 2))
        else:
            gmin = min(gmin, round(regulatory_low - abs(regulatory_low) * 0.2, 2))

    # If everything is non-negative (a common case for CEMS / flow), don't
    # let the gauge dip into negatives — operators read 0 as the floor. But if
    # the actual readings go negative (e.g. draft/pressure tags), keep the
    # negative floor so real values aren't clipped to 0 on the needle.
    data_min = min(values) if values else 0.0
    if (
        (regulatory_low is None or regulatory_low >= 0)
        and (low is None or low >= 0)
        and data_min >= 0
    ):
        gmin = max(gmin, 0.0)

    if gmax <= gmin:
        gmax = gmin + 1.0

    return low, high, gmin, gmax


def compute_gauge_zones(
    gmin: float,
    gmax: float,
    low_oper: Optional[float],
    high_oper: Optional[float],
    low_limit: Optional[float],
    high_limit: Optional[float],
    yellow_margin_pct: float = 15.0,
) -> Dict[str, object]:
    """Build a dash_daq.Gauge `color` dict with green / yellow / red zones.

    Logic (upper-bound focus, which matches CEMS reporting):
        - red:    [highest threshold, gmax]
        - yellow: between operational and regulatory limits, OR within
                  ``yellow_margin_pct`` of the threshold when only one is set
        - green:  everything below yellow

    For two-sided ranges (flow with low+high), the high side drives the
    primary zones since that is the user-visible compliance concern. The
    low side, if present, narrows the green band's lower bound.

    Returns a dict suitable for ``daq.Gauge(color=...)``. If no usable
    thresholds are present, returns a single-color (green) range so the
    gauge still renders cleanly.
    """
    high_oper_f = _to_float(high_oper)
    high_limit_f = _to_float(high_limit)
    low_oper_f = _to_float(low_oper)
    low_limit_f = _to_float(low_limit)

    margin = max(0.0, min(yellow_margin_pct, 50.0)) / 100.0

    # Determine the upper red/yellow boundaries.
    if high_oper_f is not None and high_limit_f is not None:
        yellow_lo = min(high_oper_f, high_limit_f)
        red_lo = max(high_oper_f, high_limit_f)
    elif high_limit_f is not None:
        yellow_lo = high_limit_f * (1.0 - margin) if high_limit_f >= 0 else high_limit_f * (1.0 + margin)
        red_lo = high_limit_f
    elif high_oper_f is not None:
        yellow_lo = high_oper_f * (1.0 - margin) if high_oper_f >= 0 else high_oper_f * (1.0 + margin)
        red_lo = high_oper_f
    else:
        # No upper bound — render full green.
        return {
            "gradient": False,
            "ranges": {COLOR_GOOD: [gmin, gmax]},
        }

    # Clamp boundaries inside [gmin, gmax] and keep monotonic.
    green_lo = gmin
    # If a low limit exists, the green band starts just above it.
    if low_oper_f is not None:
        green_lo = max(green_lo, low_oper_f)
    elif low_limit_f is not None:
        green_lo = max(green_lo, low_limit_f)

    yellow_lo = max(min(yellow_lo, gmax), green_lo)
    red_lo = max(min(red_lo, gmax), yellow_lo)

    ranges: Dict[str, List[float]] = {}
    if yellow_lo > green_lo:
        ranges[COLOR_GOOD] = [round(green_lo, 4), round(yellow_lo, 4)]
    if red_lo > yellow_lo:
        ranges[COLOR_WARNING] = [round(yellow_lo, 4), round(red_lo, 4)]
    if gmax > red_lo:
        ranges[COLOR_BAD] = [round(red_lo, 4), round(gmax, 4)]

    if not ranges:
        ranges[COLOR_GOOD] = [round(gmin, 4), round(gmax, 4)]

    return {"gradient": False, "ranges": ranges}


# ============================================================================
#  DATAFRAME EXTRACTION & ANALYSIS -- stats, quality metrics, compliance
# ============================================================================


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
) -> Dict[str, Tuple[float, float, datetime, datetime, int]]:
    """
    From hourly_df, extract for each tag the last full hour:
        {tag: (avg_value, avg_lb_hr, hour_start, hour_end, sample_count)}
    Any row with sample_count <= 0 is treated as "no data".
    """
    result: Dict[str, Tuple[float, float, datetime, datetime, int]] = {}
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
                result[str(tag)] = (float("nan"), float("nan"), hs, he, 0)
                continue
            try:
                avg = float(row["avg_value"])
            except Exception:
                avg = float("nan")
            try:
                avg_lb_hr = float(row.get("avg_lb_hr"))
            except Exception:
                avg_lb_hr = float("nan")
            result[str(tag)] = (avg, avg_lb_hr, hs, he, cnt)
    except Exception:
        pass

    return result


def extract_latest_rolling_12hr(
    rolling_df: pd.DataFrame,
) -> Dict[str, Tuple[float, float, datetime, datetime, int]]:
    """
    From rolling_df, extract the latest 12-hour window per tag:
        {tag: (avg_value, avg_lb_hr, window_start, window_end, hours_count)}
    Any row with hours_count <= 0 is treated as "no data".
    """
    result: Dict[str, Tuple[float, float, datetime, datetime, int]] = {}
    if rolling_df.empty or "tag" not in rolling_df.columns:
        return result

    try:
        df = rolling_df.copy()
        df = df.sort_values("window_end")
        for tag, group in df.groupby("tag"):
            row = group.iloc[-1]
            ws = row.get("window_start")
            we = row.get("window_end")
            try:
                cnt = int(row.get("hours_count", 0))
            except Exception:
                cnt = 0
            if cnt <= 0:
                result[str(tag)] = (float("nan"), float("nan"), ws, we, 0)
                continue
            try:
                avg = float(row.get("avg_value"))
            except Exception:
                avg = float("nan")
            try:
                avg_lb_hr = float(row.get("avg_lb_hr"))
            except Exception:
                avg_lb_hr = float("nan")
            result[str(tag)] = (avg, avg_lb_hr, ws, we, cnt)
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


def _month_bounds(now: datetime) -> Tuple[datetime, datetime]:
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def compute_exceedance_minutes(
    events_df: pd.DataFrame, window_start: datetime, window_end: datetime
) -> float:
    if events_df.empty:
        return 0.0
    total_sec = 0.0
    for _, row in events_df.iterrows():
        start = row.get("start_time")
        end = row.get("end_time")
        if pd.isna(start) or pd.isna(end):
            continue
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        total_sec += _overlap_seconds(start_dt, end_dt, window_start, window_end)
    return total_sec / 60.0


def _sample_within_limits(val: float, low: Optional[float], high: Optional[float]) -> bool:
    if low is not None and val < low:
        return False
    if high is not None and val > high:
        return False
    return True


def compute_within_limit_percent(
    raw_df: pd.DataFrame, thresholds: Dict[str, Dict[str, object]], window: timedelta
) -> float:
    if raw_df.empty or "timestamp" not in raw_df.columns:
        return 0.0

    cutoff = datetime.now() - window
    try:
        window_df = raw_df[raw_df["timestamp"] >= cutoff]
    except Exception:
        window_df = raw_df

    if window_df.empty:
        return 0.0

    # Only OK samples count.
    if "ok" in window_df.columns:
        window_df = window_df[window_df["ok"].fillna(False).astype(bool)]
    if window_df.empty:
        return 0.0

    # Vectorized: this runs over the full window (up to ~30 days of 7s samples,
    # millions of rows), so a per-row iterrows() loop is far too slow. Map each
    # row's tag to its configured limits and compare with array ops.
    vals = pd.to_numeric(window_df.get("value_num"), errors="coerce")
    valid = vals.notna()
    vals = vals[valid]
    if vals.empty:
        return 0.0
    tags = window_df["tag"].astype(str)[valid]

    low_map = {
        t: _to_float(e.get("low_limit"))
        for t, e in thresholds.items() if isinstance(e, dict)
    }
    high_map = {
        t: _to_float(e.get("high_limit"))
        for t, e in thresholds.items() if isinstance(e, dict)
    }
    low_arr = pd.to_numeric(tags.map(low_map), errors="coerce")
    high_arr = pd.to_numeric(tags.map(high_map), errors="coerce")

    below = low_arr.notna() & (vals < low_arr)
    above = high_arr.notna() & (vals > high_arr)
    out_of_limits = int((below | above).sum())

    total = int(len(vals))
    within = total - out_of_limits
    return round((within / float(total)) * 100.0, 1)


def compute_compliance_summary(
    raw_df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, object]],
    events_df: pd.DataFrame,
) -> Dict[str, float]:
    now = datetime.now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    month_start, month_end = _month_bounds(now)

    today_minutes = compute_exceedance_minutes(events_df, day_start, day_end)
    month_minutes = compute_exceedance_minutes(events_df, month_start, month_end)

    # The "within limits" percentages span 24 h and 30 d, but the passed raw_df
    # is only the latest single day -- the trailing 24 h already reaches into
    # yesterday, and 30 d obviously needs a month. Pull enough history so both
    # windows cover their full span instead of just today's samples. Only do the
    # heavy multi-day raw load when at least one tag actually has a regulatory
    # limit configured -- with no limits every sample is trivially "within", so
    # the result is 100% regardless and there's no reason to read 30 days of raw.
    has_limits = any(
        isinstance(e, dict) and (e.get("low_limit") is not None or e.get("high_limit") is not None)
        for e in thresholds.values()
    )
    history = raw_df
    if has_limits:
        try:
            loaded = load_raw_history(max_days=31)
            if loaded is not None and not loaded.empty:
                history = loaded
        except Exception:
            pass

    pct_24h = compute_within_limit_percent(history, thresholds, timedelta(hours=24))
    pct_30d = compute_within_limit_percent(history, thresholds, timedelta(days=30))

    return {
        "today_minutes": today_minutes,
        "month_minutes": month_minutes,
        "pct_24h": pct_24h,
        "pct_30d": pct_30d,
    }


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


# ============================================================================
#  TAG DISCOVERY & MATCHING -- find CEMS / flow tags by alias or name
# ============================================================================


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _match_cems_metric(value: Optional[str], metric: str) -> bool:
    if not value:
        return False
    normalized = _normalize_label(value)
    return "cems" in normalized and metric in normalized


def _match_flow_metric(value: Optional[str]) -> bool:
    if not value:
        return False
    normalized = _normalize_label(value)
    return "flow" in normalized


def find_cems_tag(
    metric: str,
    tags: List[str],
    thresholds: Dict[str, Dict[str, object]],
    alias_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    for tag in tags:
        entry = thresholds.get(tag, {}) if isinstance(thresholds.get(tag, {}), dict) else {}
        alias = entry.get("alias") if isinstance(entry.get("alias"), str) else ""
        if _match_cems_metric(alias, metric):
            return tag
    if alias_map:
        for tag in tags:
            alias = alias_map.get(tag, "")
            if _match_cems_metric(alias, metric):
                return tag
    for tag in tags:
        if _match_cems_metric(tag, metric):
            return tag
    return None


def find_flow_tag(
    tags: List[str],
    thresholds: Dict[str, Dict[str, object]],
    alias_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    for tag in tags:
        entry = thresholds.get(tag, {}) if isinstance(thresholds.get(tag, {}), dict) else {}
        alias = entry.get("alias") if isinstance(entry.get("alias"), str) else ""
        if _match_flow_metric(alias):
            return tag
    if alias_map:
        for tag in tags:
            alias = alias_map.get(tag, "")
            if _match_flow_metric(alias):
                return tag
    for tag in tags:
        if _match_flow_metric(tag):
            return tag
    return None


# ============================================================================
#  GAUGE CARD BUILDERS -- Flow, CEMS (ppm / lb-hr), Processing Time
# ============================================================================


def build_cems_card(
    label: str,
    tag: Optional[str],
    current_hour_lb_hr: Dict[str, float],
    current_hour_avg: Dict[str, float],
    rolling_12hr_stats: Dict[str, Tuple[float, float, datetime, datetime, int]],
    thresholds: Dict[str, Dict[str, object]],
    alias_map: Optional[Dict[str, str]] = None,
    units_label: str = "ppm",
    value_source: str = "avg_value",
) -> html.Div:
    entry = thresholds.get(tag, {}) if tag and isinstance(thresholds.get(tag, {}), dict) else {}
    alias = entry.get("alias") if isinstance(entry.get("alias"), str) else None
    if not alias and tag and alias_map:
        alias = alias_map.get(tag)

    low_oper = entry.get("low_oper")
    high_oper = entry.get("high_oper")
    low_limit = entry.get("low_limit")
    high_limit = entry.get("high_limit")

    if value_source == "avg_lb_hr":
        value = current_hour_lb_hr.get(tag, float("nan")) if tag else float("nan")
    else:
        value = current_hour_avg.get(tag, float("nan")) if tag else float("nan")
    rolling_entry = rolling_12hr_stats.get(
        tag, (float("nan"), float("nan"), None, None, 0)
    )
    rolling_avg_value, rolling_avg_lb_hr, ws, we, rolling_count = rolling_entry
    rolling_avg = rolling_avg_lb_hr if value_source == "avg_lb_hr" else rolling_avg_value

    low_for_range = low_oper if low_oper is not None else low_limit
    high_for_range = high_oper if high_oper is not None else high_limit
    low_eff, high_eff, gmin, gmax = compute_gauge_range(
        low_for_range, high_for_range, [value, rolling_avg],
        regulatory_high=_to_float(high_limit),
        regulatory_low=_to_float(low_limit),
    )
    low_for_class = low_oper if low_oper is not None else low_eff
    high_for_class = high_oper if high_oper is not None else high_eff

    status = classify_value(value, low_for_class, high_for_class)
    border_color = status_color(status)
    zone_color = compute_gauge_zones(
        gmin, gmax,
        low_oper=low_oper, high_oper=high_oper,
        low_limit=low_limit, high_limit=high_limit,
    )

    if (
        rolling_avg == rolling_avg
        and ws is not None
        and we is not None
        and rolling_count > 0
    ):
        try:
            ws_s = ws.strftime("%Y-%m-%d %H:%M")
            we_s = we.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ws_s = str(ws)
            we_s = str(we)
        rolling_value_text = f"{rolling_avg:.2f} {units_label}"
        rolling_text = f"Rolling 12h {ws_s}–{we_s}: {rolling_value_text} ({rolling_count} hrs)"
    else:
        rolling_text = "Rolling 12h: no data"

    # Show permit limit in subtitle when set
    header = alias or label
    subtitle_bits = []
    if tag:
        subtitle_bits.append(f"Tag: {tag}")
    else:
        subtitle_bits.append("No matching tag found yet")
    subtitle_bits.append(f"Units: {units_label}")
    if high_limit is not None:
        subtitle_bits.append(f"Permit: {high_limit} {units_label}")
    subtitle = " • ".join(subtitle_bits)

    card_style = {
        **CARD_STYLE,
        "borderTop": f"3px solid {border_color}",
    }

    return html.Div(
        style=card_style,
        children=[
            html.Div(header, style=CARD_HEADER_STYLE),
            html.Div(subtitle, style=CARD_SUBTITLE_STYLE),
            html.Div(
                style=GAUGE_CONTAINER_STYLE,
                children=[
                    daq.Gauge(
                        id={"type": "cems-gauge", "tag": tag or label},
                        min=gmin,
                        max=gmax,
                        # No current-hour data: rest the needle at the floor but
                        # do NOT print a number or color it, so a downed analyzer
                        # reads as "no data" instead of a confident 0.0 in green.
                        value=value if value == value else gmin,
                        showCurrentValue=value == value,
                        color=zone_color if value == value else COLOR_TEXT_SUBTLE,
                        label={
                            "label": (
                                f"Current hour ({units_label})"
                                if value == value else "Current hour: no data"
                            ),
                            "style": {"fontSize": "12px"},
                        },
                        size=220,
                        units=units_label,
                    ),
                ],
            ),
            html.Div(rolling_text, style=CARD_FOOTER_STYLE),
        ],
    )


def build_flow_card(
    label: str,
    tag: Optional[str],
    current_hour_avg: Dict[str, float],
    rolling_12hr_stats: Dict[str, Tuple[float, float, datetime, datetime, int]],
    thresholds: Dict[str, Dict[str, object]],
    alias_map: Optional[Dict[str, str]] = None,
) -> html.Div:
    entry = thresholds.get(tag, {}) if tag and isinstance(thresholds.get(tag, {}), dict) else {}
    alias = entry.get("alias") if isinstance(entry.get("alias"), str) else None
    if not alias and tag and alias_map:
        alias = alias_map.get(tag)

    units = entry.get("units") if isinstance(entry.get("units"), str) else ""
    units_label = units or "scfm"

    low_oper = entry.get("low_oper")
    high_oper = entry.get("high_oper")
    low_limit = entry.get("low_limit")
    high_limit = entry.get("high_limit")

    value = current_hour_avg.get(tag, float("nan")) if tag else float("nan")
    rolling_entry = rolling_12hr_stats.get(tag, (float("nan"), float("nan"), None, None, 0))
    rolling_avg = rolling_entry[0]
    _, _, ws, we, rolling_count = rolling_entry

    low_for_range = low_oper if low_oper is not None else low_limit
    high_for_range = high_oper if high_oper is not None else high_limit
    low_eff, high_eff, gmin, gmax = compute_gauge_range(
        low_for_range, high_for_range, [value, rolling_avg],
        regulatory_high=_to_float(high_limit),
        regulatory_low=_to_float(low_limit),
    )
    low_for_class = low_oper if low_oper is not None else low_eff
    high_for_class = high_oper if high_oper is not None else high_eff

    status = classify_value(value, low_for_class, high_for_class)
    border_color = status_color(status)
    zone_color = compute_gauge_zones(
        gmin, gmax,
        low_oper=low_oper, high_oper=high_oper,
        low_limit=low_limit, high_limit=high_limit,
    )

    if (
        rolling_avg == rolling_avg
        and ws is not None
        and we is not None
        and rolling_count > 0
    ):
        try:
            ws_s = ws.strftime("%Y-%m-%d %H:%M")
            we_s = we.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ws_s = str(ws)
            we_s = str(we)
        rolling_text = (
            f"Rolling 12h {ws_s}–{we_s}: {rolling_avg:.2f} {units_label} "
            f"({rolling_count} hrs)"
        )
    else:
        rolling_text = "Rolling 12h: no data"

    header = alias or label
    subtitle_bits = []
    if tag:
        subtitle_bits.append(f"Tag: {tag}")
    else:
        subtitle_bits.append("No matching tag found yet")
    subtitle_bits.append(f"Units: {units_label}")
    subtitle = " • ".join(subtitle_bits)

    card_style = {
        **CARD_STYLE,
        "borderTop": f"3px solid {border_color}",
    }

    return html.Div(
        style=card_style,
        children=[
            html.Div(header, style=CARD_HEADER_STYLE),
            html.Div(subtitle, style=CARD_SUBTITLE_STYLE),
            html.Div(
                style=GAUGE_CONTAINER_STYLE,
                children=[
                    daq.Gauge(
                        id={"type": "flow-gauge", "tag": tag or label},
                        min=gmin,
                        max=gmax,
                        # No current-hour data: rest the needle at the floor but
                        # do NOT print a number or color it (see CEMS card).
                        value=value if value == value else gmin,
                        showCurrentValue=value == value,
                        color=zone_color if value == value else COLOR_TEXT_SUBTLE,
                        label={
                            "label": (
                                "Current hourly avg"
                                if value == value else "Current hourly avg: no data"
                            ),
                            "style": {"fontSize": "12px"},
                        },
                        size=220,
                        units=units_label,
                    ),
                ],
            ),
            html.Div(rolling_text, style=CARD_FOOTER_STYLE),
        ],
    )


def build_processing_time_card(
    current_hour_minutes: float,
    today_total_minutes: float,
    all_time_total_minutes: Optional[float] = None,
) -> html.Div:
    """
    Build a gauge card showing processing time for the current hour.
    Gauge runs 0–60 minutes. Below the gauge: today's total (prominent),
    then all-time total across every raw_data_*.csv (secondary).
    """
    # Gauge range is fixed: 0 to 60 minutes (one full hour). The full red /
    # yellow / green bands give an at-a-glance read of how much of the hour
    # was spent processing.
    gmin = 0
    gmax = 60
    zone_color = {
        "gradient": False,
        "ranges": {
            COLOR_BAD: [0, 15],       # very little processing this hour
            COLOR_WARNING: [15, 45],  # partial hour
            COLOR_GOOD: [45, 60],     # most of the hour
        },
    }

    value = current_hour_minutes
    has_value = isinstance(value, (int, float)) and value == value

    if has_value:
        value = min(max(value, 0), 60)
        if value >= 45:
            border_color = COLOR_GOOD
        elif value >= 15:
            border_color = COLOR_WARNING
        else:
            border_color = COLOR_BAD
        value_label = f"{value:.1f}"
    else:
        border_color = COLOR_TEXT_SUBTLE
        value_label = "—"
        value = 0

    # Today total (prominent)
    has_today = isinstance(today_total_minutes, (int, float)) and today_total_minutes == today_total_minutes
    if has_today:
        today_hrs = today_total_minutes / 60.0
        today_text = f"Today: {today_hrs:.2f} hrs ({today_total_minutes:.1f} min)"
    else:
        today_text = "Today: no data"

    # All-time total (secondary)
    has_all_time = (
        isinstance(all_time_total_minutes, (int, float))
        and all_time_total_minutes == all_time_total_minutes
    )
    if has_all_time:
        all_time_hrs = all_time_total_minutes / 60.0
        all_time_text = f"All time: {all_time_hrs:.2f} hrs"
    else:
        all_time_text = "All time: no data"

    today_style = {
        "fontSize": "14px",
        "fontWeight": "600",
        "color": COLOR_TEXT_PRIMARY,
        "borderTop": f"1px solid {COLOR_BORDER}",
        "paddingTop": "8px",
        "marginTop": "2px",
        "textAlign": "center",
    }
    all_time_style = {
        "fontSize": "11px",
        "color": COLOR_TEXT_MUTED,
        "paddingTop": "4px",
        "textAlign": "center",
    }

    card_style = {
        **CARD_STYLE,
        "borderTop": f"3px solid {border_color}",
    }

    return html.Div(
        style=card_style,
        children=[
            html.Div("Processing Time", style=CARD_HEADER_STYLE),
            html.Div(
                "Time in Processing state (state 3) this hour",
                style=CARD_SUBTITLE_STYLE,
            ),
            html.Div(
                style=GAUGE_CONTAINER_STYLE,
                children=[
                    daq.Gauge(
                        id="processing-time-gauge",
                        min=gmin,
                        max=gmax,
                        value=value,
                        showCurrentValue=True,
                        color=zone_color,
                        label={"label": "Minutes this hour", "style": {"fontSize": "12px"}},
                        size=220,
                        units="min",
                    ),
                ],
            ),
            html.Div(today_text, style=today_style),
            html.Div(all_time_text, style=all_time_style),
        ],
    )


def build_feed_status_card(
    feed_status: Dict[str, object],
    processing_running: bool,
) -> html.Div:
    """Build a card showing feed system and processing status side-by-side."""

    feed_running = feed_status.get("running")
    feed_method = feed_status.get("method", "none")
    feed_detail = str(feed_status.get("detail", ""))
    feed_rate = feed_status.get("feed_rate")
    weight_units = str(feed_status.get("weight_units", "") or "")

    # --- Feed system indicator ---
    if feed_running is True:
        feed_color = COLOR_GOOD
        feed_label = "FEEDING"
    elif feed_running is False:
        feed_color = COLOR_TEXT_SUBTLE
        feed_label = "IDLE"
    else:
        feed_color = COLOR_TEXT_SUBTLE
        feed_label = "UNKNOWN"

    method_labels = {"weight": "Weight-Based", "tag": "Tag-Based", "none": "Not Configured"}
    method_text = method_labels.get(feed_method, feed_method)

    feed_children = [
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px"},
            children=[
                html.Div(style={
                    "width": "10px",
                    "height": "10px",
                    "borderRadius": "50%",
                    "backgroundColor": feed_color,
                    "boxShadow": f"0 0 6px {feed_color}" if feed_running else "none",
                    "flexShrink": "0",
                }),
                html.Div(feed_label, style={
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "color": feed_color,
                }),
            ],
        ),
        html.Div(f"Detection: {method_text}", style={"fontSize": "11px", "color": COLOR_TEXT_MUTED}),
    ]

    if feed_detail and feed_detail not in ("ON", "OFF"):
        feed_children.append(
            html.Div(feed_detail, style={"fontSize": "12px", "color": COLOR_TEXT_PRIMARY})
        )

    if feed_rate is not None and feed_rate > 0:
        units_label = weight_units or "units"
        feed_children.append(
            html.Div(
                f"Rate: {feed_rate:.1f} {units_label}/min",
                style={"fontSize": "14px", "fontWeight": "600", "color": COLOR_TEXT_PRIMARY},
            )
        )

    # --- Processing indicator ---
    if processing_running:
        proc_color = COLOR_GOOD
        proc_label = "PROCESSING"
    else:
        proc_color = COLOR_TEXT_SUBTLE
        proc_label = "NOT PROCESSING"

    proc_children = [
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px"},
            children=[
                html.Div(style={
                    "width": "10px",
                    "height": "10px",
                    "borderRadius": "50%",
                    "backgroundColor": proc_color,
                    "boxShadow": f"0 0 6px {proc_color}" if processing_running else "none",
                    "flexShrink": "0",
                }),
                html.Div(proc_label, style={
                    "fontSize": "18px",
                    "fontWeight": "700",
                    "color": proc_color,
                }),
            ],
        ),
        html.Div("Machine state 3 = Processing", style={"fontSize": "11px", "color": COLOR_TEXT_MUTED}),
    ]

    # Pick card border color: green if both running, yellow if only one, gray if neither
    if feed_running and processing_running:
        border_color = COLOR_GOOD
    elif feed_running or processing_running:
        border_color = COLOR_WARNING
    else:
        border_color = COLOR_TEXT_SUBTLE

    card_style = {
        **CARD_STYLE,
        "borderTop": f"3px solid {border_color}",
    }

    return html.Div(
        style=card_style,
        children=[
            html.Div("System Status", style=CARD_HEADER_STYLE),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "16px",
                },
                children=[
                    # Feed column
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                        children=feed_children,
                    ),
                    # Processing column
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                            "borderLeft": f"1px solid {COLOR_BORDER}",
                            "paddingLeft": "16px",
                        },
                        children=proc_children,
                    ),
                ],
            ),
        ],
    )


def build_cems_uptime_card(
    month_stats: Dict[str, object],
    quarter_stats: Dict[str, object],
    required_pct: float = 90.0,
) -> html.Div:
    """Build a card showing CEMS data availability for the current month and quarter."""

    def _uptime_section(label: str, stats: Dict[str, object]) -> html.Div:
        pct = float(stats.get("uptime_pct", 0))
        op_hrs = int(stats.get("operating_hours", 0))
        valid_hrs = int(stats.get("valid_cems_hours", 0))
        missing_hrs = int(stats.get("missing_hours", 0))

        if op_hrs == 0:
            color = COLOR_TEXT_SUBTLE
            status_text = "No operating hours"
        elif pct >= required_pct:
            color = COLOR_GOOD
            status_text = f"{pct:.1f}%"
        elif pct >= required_pct - 5:
            color = COLOR_WARNING
            status_text = f"{pct:.1f}%"
        else:
            color = COLOR_BAD
            status_text = f"{pct:.1f}%"

        return html.Div(
            style={"display": "flex", "flexDirection": "column", "gap": "6px"},
            children=[
                html.Div(label, style={
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "color": COLOR_TEXT_MUTED,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                }),
                html.Div(
                    style={"display": "flex", "alignItems": "baseline", "gap": "6px"},
                    children=[
                        html.Div(status_text, style={
                            "fontSize": "28px",
                            "fontWeight": "800",
                            "color": color,
                            "lineHeight": "1",
                        }),
                        html.Div(
                            f"of {required_pct:.0f}% required",
                            style={"fontSize": "11px", "color": COLOR_TEXT_MUTED},
                        ) if op_hrs > 0 else html.Div(),
                    ],
                ),
                # Progress bar
                html.Div(
                    style={
                        "height": "6px",
                        "backgroundColor": COLOR_SURFACE_ALT,
                        "borderRadius": "3px",
                        "overflow": "hidden",
                        "position": "relative",
                    },
                    children=[
                        html.Div(style={
                            "height": "100%",
                            "width": f"{min(pct, 100):.1f}%",
                            "backgroundColor": color,
                            "borderRadius": "3px",
                            "transition": "width 0.5s ease",
                        }),
                    ],
                ) if op_hrs > 0 else html.Div(),
                html.Div(
                    f"{valid_hrs} valid / {op_hrs} operating hrs"
                    + (f" ({missing_hrs} missing)" if missing_hrs > 0 else ""),
                    style={"fontSize": "11px", "color": COLOR_TEXT_MUTED},
                ) if op_hrs > 0 else html.Div(),
            ],
        )

    # Card border color based on worst-case of the two periods
    month_pct = float(month_stats.get("uptime_pct", 0))
    month_op = int(month_stats.get("operating_hours", 0))
    if month_op == 0:
        border_color = COLOR_TEXT_SUBTLE
    elif month_pct >= required_pct:
        border_color = COLOR_GOOD
    elif month_pct >= required_pct - 5:
        border_color = COLOR_WARNING
    else:
        border_color = COLOR_BAD

    card_style = {**CARD_STYLE, "borderTop": f"3px solid {border_color}"}

    return html.Div(
        style=card_style,
        children=[
            html.Div("CEMS Data Availability", style=CARD_HEADER_STYLE),
            html.Div(
                "Monitor uptime vs. operating hours (requires machine state tag)",
                style=CARD_SUBTITLE_STYLE,
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                },
                children=[
                    _uptime_section("This Month", month_stats),
                    html.Div(
                        style={
                            "borderLeft": f"1px solid {COLOR_BORDER}",
                            "paddingLeft": "20px",
                        },
                        children=[_uptime_section("This Quarter", quarter_stats)],
                    ),
                ],
            ),
        ],
    )


# ============================================================================
#  DASHBOARD COMPONENT BUILDERS -- health, quality, compliance, stats
# ============================================================================


def build_system_health_card(health: Dict[str, object]) -> html.Div:
    status = str(health.get("status", "Unknown"))
    status_reason = str(health.get("status_reason", ""))
    status_upper = status.lower()
    if "critical" in status_upper:
        color = COLOR_BAD
    elif "degraded" in status_upper:
        color = COLOR_WARNING
    elif "healthy" in status_upper:
        color = COLOR_GOOD
    else:
        color = COLOR_TEXT_SUBTLE

    details = []
    for label, key in [
        ("Last poll", "last_poll_success_ts"),
        ("Last error", "last_poll_error_ts"),
        ("Errors (1h)", "error_count_last_hour"),
        ("Disk free (GB)", "disk_free_GB"),
        ("Log dir (GB)", "log_dir_size_GB"),
    ]:
        val = health.get(key)
        details.append(
            html.Div(
                [
                    html.Span(f"{label}: ", style={"color": COLOR_TEXT_MUTED}),
                    html.Span("—" if val is None else str(val)),
                ],
                style={"fontSize": "12px"},
            )
        )

    card_style = {
        **CARD_STYLE,
        "borderTop": f"3px solid {color}",
    }

    return html.Div(
        style=card_style,
        children=[
            html.Div("System Health", style=CARD_HEADER_STYLE),
            html.Div(
                status,
                style={"color": color, "fontSize": "22px", "fontWeight": "700", "marginTop": "2px"},
            ),
            html.Div(
                status_reason,
                style={"fontSize": "12px", "color": COLOR_TEXT_MUTED},
            ),
            html.Div(
                details,
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "3px",
                    "borderTop": f"1px solid {COLOR_BORDER}",
                    "paddingTop": "8px",
                    "marginTop": "4px",
                },
            ),
        ],
    )


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
            html.Div("Data Quality (last 24h)", style=CARD_HEADER_STYLE),
            html.Div(
                "QA flags drive valid %; DATA_GAP events summarize missing runs.",
                style=CARD_SUBTITLE_STYLE,
            ),
            table,
        ],
    )


def build_stat_tile(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        style={
            "backgroundColor": COLOR_SURFACE_ALT,
            "padding": "14px 16px",
            "borderRadius": "12px",
            "minWidth": "220px",
            "color": COLOR_TEXT_PRIMARY,
            "border": f"1px solid {COLOR_BORDER}",
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "color": COLOR_TEXT_MUTED, "fontWeight": "500"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "700", "marginTop": "6px"}),
            html.Div(
                subtitle,
                style={"fontSize": "11px", "color": COLOR_TEXT_SUBTLE, "marginTop": "3px"},
            ),
        ],
    )


def build_exceedance_table(events_df: pd.DataFrame, limit: int = 20):
    if events_df.empty:
        return html.Div(
            "No exceedance events found in the current window.",
            style={"color": COLOR_TEXT_MUTED, "fontSize": "11px"},
        )

    table_df = events_df.copy()
    table_df = table_df.sort_values("start_time", ascending=False).head(limit)
    table_df["start_time"] = table_df["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    table_df["end_time"] = table_df["end_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    table_df["duration_min"] = (table_df["duration_sec"].astype(float) / 60.0).round(1)

    return dash_table.DataTable(
        data=table_df[
            [
                "tag",
                "start_time",
                "end_time",
                "duration_min",
                "max_value",
                "avg_value_over_event",
                "limit_value",
            ]
        ].to_dict("records"),
        columns=[
            {"name": "Tag", "id": "tag"},
            {"name": "Start", "id": "start_time"},
            {"name": "End", "id": "end_time"},
            {"name": "Duration (min)", "id": "duration_min"},
            {"name": "Max", "id": "max_value"},
            {"name": "Avg", "id": "avg_value_over_event"},
            {"name": "Limit", "id": "limit_value"},
        ],
        style_header={"backgroundColor": COLOR_TABLE_HEADER, "color": COLOR_TEXT_PRIMARY},
        style_cell={
            "backgroundColor": COLOR_TABLE_CELL,
            "color": COLOR_TEXT_PRIMARY,
            "fontSize": "11px",
            "padding": "6px",
            "textAlign": "left",
        },
        page_size=limit,
    )


def _format_threshold_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and value != value:
            return ""
    except Exception:
        return ""
    return str(value)


def build_compliance_rules_card() -> html.Div:
    rules = html.Ul(
        style={
            "margin": 0,
            "paddingLeft": "18px",
            "fontSize": "11px",
            "color": COLOR_TEXT_SUBTLE,
        },
        children=[
            html.Li("Regulatory compliance uses low/high limits from thresholds.json."),
            html.Li("Operational thresholds only drive gauge warnings and are not compliance limits."),
            html.Li("Exceedances are logged when samples fall outside compliance limits."),
            html.Li("Within-limit percentages only count samples with QA flag = OK."),
            html.Li("Edit thresholds below to keep limits aligned with permit requirements."),
        ],
    )

    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(
                "Compliance Rules & Requirements",
                style={"fontWeight": "600", "fontSize": "13px"},
            ),
            rules,
        ],
    )


def build_compliance_thresholds_table(
    thresholds: Dict[str, Dict[str, object]], tags: List[str]
) -> html.Div:
    rows = []
    for tag in sorted(tags, key=str):
        entry = thresholds.get(tag, {}) if isinstance(thresholds.get(tag, {}), dict) else {}
        rows.append(
            {
                "tag": tag,
                "alias": entry.get("alias", "") or "",
                "units": entry.get("units", "") or "",
                "low_oper": _format_threshold_cell(entry.get("low_oper")),
                "high_oper": _format_threshold_cell(entry.get("high_oper")),
                "low_limit": _format_threshold_cell(entry.get("low_limit")),
                "high_limit": _format_threshold_cell(entry.get("high_limit")),
            }
        )

    table = dash_table.DataTable(
        id="compliance-thresholds-table",
        data=rows,
        columns=[
            {"name": "Tag", "id": "tag", "presentation": "input"},
            {"name": "Alias", "id": "alias", "presentation": "input"},
            {"name": "Units", "id": "units", "presentation": "input"},
            {"name": "Low Oper", "id": "low_oper", "presentation": "input"},
            {"name": "High Oper", "id": "high_oper", "presentation": "input"},
            {"name": "Low Limit", "id": "low_limit", "presentation": "input"},
            {"name": "High Limit", "id": "high_limit", "presentation": "input"},
        ],
        editable=True,
        style_header={"backgroundColor": COLOR_TABLE_HEADER, "color": COLOR_TEXT_PRIMARY},
        style_cell={
            "backgroundColor": COLOR_TABLE_CELL,
            "color": COLOR_TEXT_PRIMARY,
            "fontSize": "11px",
            "padding": "6px",
            "textAlign": "left",
        },
        style_table={"maxHeight": "300px", "overflowY": "auto"},
        page_size=12,
    )

    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(
                "Compliance Thresholds (edit to match CIP settings)",
                style={"fontWeight": "600", "fontSize": "13px"},
            ),
            html.Div(
                "Updates are saved to thresholds.json and consumed by CIP.py.",
                style={"fontSize": "11px", "color": COLOR_TEXT_MUTED},
            ),
            table,
            html.Div(
                style={"display": "flex", "gap": "8px", "alignItems": "center"},
                children=[
                    html.Button(
                        "Save Compliance Thresholds",
                        id="compliance-threshold-save-btn",
                        n_clicks=0,
                        style={"marginTop": "6px", **EXPORT_BUTTON_STYLE},
                    ),
                    html.Div(
                        id="compliance-threshold-save-status",
                        style={"fontSize": "11px", "color": COLOR_TEXT_SUBTLE},
                    ),
                ],
            ),
        ],
    )


def build_compliance_view(
    summary: Dict[str, float],
    events_df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, object]],
    tags: List[str],
) -> html.Div:
    tiles = html.Div(
        style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
        children=[
            build_stat_tile(
                "Today's exceedance minutes",
                f"{summary.get('today_minutes', 0.0):.1f} min",
            ),
            build_stat_tile(
                "Month-to-date exceedance minutes",
                f"{summary.get('month_minutes', 0.0):.1f} min",
            ),
        ],
    )

    gauges = html.Div(
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
        children=[
            daq.Gauge(
                min=0,
                max=100,
                value=summary.get("pct_24h", 0.0),
                showCurrentValue=True,
                label="Within limits (last 24h)",
                color=COLOR_GOOD,
                size=180,
            ),
            daq.Gauge(
                min=0,
                max=100,
                value=summary.get("pct_30d", 0.0),
                showCurrentValue=True,
                label="Within limits (last 30d)",
                color="#6ee7b7",
                size=180,
            ),
        ],
    )

    table = html.Div(
        children=[
            html.Div(
                "Recent exceedance events",
                style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "6px"},
            ),
            build_exceedance_table(events_df, limit=20),
        ]
    )

    rules_card = build_compliance_rules_card()
    thresholds_card = build_compliance_thresholds_table(thresholds, tags)

    return html.Div(
        style={"padding": "12px", "display": "flex", "flexDirection": "column", "gap": "12px"},
        children=[tiles, gauges, rules_card, thresholds_card, table],
    )


# ============================================================================
#  DASH APP LAYOUT
# ============================================================================

app = Dash(__name__)
app.title = "ARC Reporting"
server = app.server

app.layout = html.Div(
    style={
        "backgroundColor": COLOR_BG,
        "color": COLOR_TEXT_PRIMARY,
        "fontFamily": "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
        "padding": "20px 24px",
        "minHeight": "100vh",
    },
    children=[
        # Header
        html.Div(
            style={
                "backgroundColor": COLOR_SURFACE,
                "borderRadius": "14px",
                "padding": "16px 24px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "16px",
                "boxShadow": "0 2px 12px rgba(0,0,0,0.25)",
                "borderBottom": f"2px solid {COLOR_ACCENT}",
            },
            children=[
                html.Div(
                    children=[
                        html.Div(
                            "ARC Reporting",
                            style={
                                "fontSize": "22px",
                                "fontWeight": "800",
                                "letterSpacing": "-0.02em",
                                "color": COLOR_TEXT_PRIMARY,
                            },
                        ),
                        html.Div(
                            "Live PLC tag monitoring & hourly aggregates",
                            style={
                                "fontSize": "12px",
                                "color": COLOR_TEXT_MUTED,
                                "marginTop": "2px",
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
                                "color": COLOR_TEXT_SUBTLE,
                                "textAlign": "right",
                            },
                        ),
                        html.Div(
                            id="last-update-label",
                            style={
                                "fontSize": "12px",
                                "color": COLOR_TEXT_MUTED,
                                "textAlign": "right",
                                "fontWeight": "500",
                                "marginTop": "2px",
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
        dcc.Download(id="export-minute-download"),
        dcc.Download(id="export-hourly-download"),
        dcc.Download(id="export-rolling-download"),
        dcc.Download(id="export-range-raw-download"),
        dcc.Download(id="export-range-minute-download"),
        dcc.Download(id="export-range-hourly-download"),
        dcc.Download(id="export-range-rolling-download"),
        dcc.Download(id="export-nox-calc-raw-download"),
        dcc.Download(id="export-nox-calc-minute-download"),
        dcc.Download(id="export-nox-calc-hourly-download"),
        dcc.Download(id="export-emission-totals-download"),
        dcc.Download(id="report-today-download"),
        dcc.Download(id="report-week-download"),
        dcc.Download(id="report-month-download"),
        dcc.Download(id="report-prev-month-download"),
        dcc.Download(id="report-all-time-download"),
        dcc.Download(id="incident-report-today-download"),
        dcc.Download(id="incident-report-week-download"),
        dcc.Download(id="incident-report-month-download"),
        dcc.Download(id="incident-report-prev-month-download"),
        dcc.Download(id="incident-report-all-time-download"),
        dcc.Download(id="daily-ops-today-download"),
        dcc.Download(id="daily-ops-week-download"),
        dcc.Download(id="daily-ops-month-download"),
        dcc.Download(id="daily-ops-prev-month-download"),
        dcc.Download(id="daily-ops-all-time-download"),

        # Tabs: Overview (gauges) & Thresholds (editor)
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            colors={
                "border": "transparent",
                "primary": COLOR_ACCENT,
                "background": COLOR_SURFACE_ALT,
            },
            style={
                "borderRadius": "12px",
                "overflow": "hidden",
            },
            children=[
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[
                        html.Div(
                            id="tag-cards-container",
                            style={
                                "padding": "20px 16px",
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fill, minmax(280px, 1fr))",
                                "gap": "20px",
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Thresholds",
                    value="thresholds",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[
                        html.Div(
                            style={
                                "padding": "20px 24px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "12px",
                                "maxWidth": "480px",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            "Threshold / Limit Editor",
                                            style={
                                                "fontSize": "16px",
                                                "fontWeight": "700",
                                            },
                                        ),
                                        html.Div(
                                            "Operational thresholds drive gauge coloring. "
                                            "Regulatory limits track permit compliance.",
                                            style={
                                                "fontSize": "12px",
                                                "color": COLOR_TEXT_MUTED,
                                                "marginTop": "4px",
                                                "lineHeight": "1.5",
                                            },
                                        ),
                                    ],
                                ),
                                html.Label(
                                    "Tag:",
                                    style={"fontSize": "12px", "fontWeight": "600"},
                                ),
                                dcc.Dropdown(
                                    id="threshold-tag-dropdown",
                                    options=[],
                                    value=None,
                                    placeholder="Select a tag...",
                                ),
                                html.Label(
                                    "Alias (friendly name):",
                                    style={"fontSize": "12px", "fontWeight": "600"},
                                ),
                                dcc.Input(
                                    id="threshold-alias-input",
                                    type="text",
                                    placeholder="e.g. Kiln Temperature",
                                    style=INPUT_STYLE,
                                ),
                                html.Label(
                                    "Units:",
                                    style={"fontSize": "12px", "fontWeight": "600"},
                                ),
                                dcc.Input(
                                    id="threshold-units-input",
                                    type="text",
                                    placeholder="e.g. ppm, lb/hr, %",
                                    style=INPUT_STYLE,
                                ),
                                html.Label(
                                    "Operational thresholds (warning band):",
                                    style={"fontSize": "12px", "fontWeight": "600"},
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "10px"},
                                    children=[
                                        dcc.Input(
                                            id="threshold-low-input",
                                            type="number",
                                            placeholder="Low",
                                            style={**INPUT_STYLE, "flex": "1"},
                                        ),
                                        dcc.Input(
                                            id="threshold-high-input",
                                            type="number",
                                            placeholder="High",
                                            style={**INPUT_STYLE, "flex": "1"},
                                        ),
                                    ],
                                ),
                                html.Label(
                                    "Regulatory limits (permit/compliance):",
                                    style={"fontSize": "12px", "fontWeight": "600"},
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "10px"},
                                    children=[
                                        dcc.Input(
                                            id="threshold-low-limit-input",
                                            type="number",
                                            placeholder="Low limit",
                                            style={**INPUT_STYLE, "flex": "1"},
                                        ),
                                        dcc.Input(
                                            id="threshold-high-limit-input",
                                            type="number",
                                            placeholder="High limit",
                                            style={**INPUT_STYLE, "flex": "1"},
                                        ),
                                    ],
                                ),
                                html.Button(
                                    "Save Thresholds",
                                    id="threshold-save-btn",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "4px",
                                        **EXPORT_BUTTON_STYLE,
                                    },
                                ),
                                html.Div(
                                    id="threshold-save-status",
                                    style={"fontSize": "12px", "color": COLOR_TEXT_MUTED},
                                ),
                                html.Div(
                                    f"File: {THRESHOLDS_JSON}",
                                    style={
                                        "fontSize": "10px",
                                        "color": COLOR_TEXT_SUBTLE,
                                        "marginTop": "4px",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Exports",
                    value="exports",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[
                        html.Div(
                            style={
                                "padding": "20px 24px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "16px",
                                "maxWidth": "800px",
                            },
                            children=[
                                html.Div(
                                    style=CARD_STYLE,
                                    children=[
                                        html.Div("Data Exports (CSV)", style=EXPORT_SECTION_TITLE_STYLE),
                                        html.Div(
                                            "Download raw exports for minute, hourly, and rolling averages.",
                                            style=EXPORT_SECTION_HELP_STYLE,
                                        ),
                                        html.Div(
                                            style=EXPORT_BUTTON_ROW_STYLE,
                                            children=[
                                                html.Button(
                                                    "Minute Averages",
                                                    id="export-minute-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Hourly Averages",
                                                    id="export-hourly-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Rolling 12 Hour Averages",
                                                    id="export-rolling-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            children=html.Div(
                                                id="export-data-status",
                                                style={
                                                    "marginTop": "8px",
                                                    "fontSize": "11px",
                                                    "color": COLOR_TEXT_MUTED,
                                                },
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=CARD_STYLE,
                                    children=[
                                        html.Div("Time-Range Data Exports (CSV)", style=EXPORT_SECTION_TITLE_STYLE),
                                        html.Div(
                                            "Export raw, minute, hourly, and rolling files for a selected period. Files are copied to logs/exports before range filtering.",
                                            style=EXPORT_SECTION_HELP_STYLE,
                                        ),
                                        dcc.Dropdown(
                                            id="export-range-dropdown",
                                            options=[
                                                {"label": "Today", "value": "today"},
                                                {"label": "This Week", "value": "week"},
                                                {"label": "This Month", "value": "month"},
                                                {"label": "Previous Month", "value": "prev_month"},
                                                {"label": "All Time", "value": "all_time"},
                                            ],
                                            value="month",
                                            clearable=False,
                                            style={"marginTop": "8px", "fontSize": "12px"},
                                        ),
                                        html.Div(
                                            style=EXPORT_BUTTON_ROW_STYLE,
                                            children=[
                                                html.Button(
                                                    "Raw Data",
                                                    id="export-range-raw-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Minute Averages",
                                                    id="export-range-minute-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Hourly Averages",
                                                    id="export-range-hourly-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Rolling 12 Hour Averages",
                                                    id="export-range-rolling-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            children=html.Div(
                                                id="export-range-status",
                                                style={
                                                    "marginTop": "8px",
                                                    "fontSize": "11px",
                                                    "color": COLOR_TEXT_MUTED,
                                                },
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=CARD_STYLE,
                                    children=[
                                        html.Div(
                                            "NOx Manual Calculation Worksheet (CSV)",
                                            style=EXPORT_SECTION_TITLE_STYLE,
                                        ),
                                        html.Div(
                                            "NOx-only export with ppm and gas flow on the same row "
                                            "for manual EPA Method 19 verification. Includes the "
                                            "MW (46.0), molar volume (385.3 dscf/lb-mol), and the "
                                            "computed lb/hr alongside the inputs. Uses the "
                                            "time-range selected above.",
                                            style=EXPORT_SECTION_HELP_STYLE,
                                        ),
                                        html.Div(
                                            style=EXPORT_BUTTON_ROW_STYLE,
                                            children=[
                                                html.Button(
                                                    "Raw (per sample)",
                                                    id="export-nox-calc-raw-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Minute Averages",
                                                    id="export-nox-calc-minute-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Hourly Averages",
                                                    id="export-nox-calc-hourly-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            children=html.Div(
                                                id="export-nox-calc-status",
                                                style={
                                                    "marginTop": "8px",
                                                    "fontSize": "11px",
                                                    "color": COLOR_TEXT_MUTED,
                                                },
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=CARD_STYLE,
                                    children=[
                                        html.Div(
                                            "Emission Totals (CSV)",
                                            style=EXPORT_SECTION_TITLE_STYLE,
                                        ),
                                        html.Div(
                                            "NOx and CO totals in lb for the selected time range, "
                                            "calendar year-to-date, and trailing 12 months. "
                                            "Each total is recomputed from the per-hour ppm and "
                                            "flow averages using the simplified EPA Method 19 "
                                            "formula, so historical hours that were logged with "
                                            "the older double-O2-correction formula contribute "
                                            "the correct mass.",
                                            style=EXPORT_SECTION_HELP_STYLE,
                                        ),
                                        html.Div(
                                            style=EXPORT_BUTTON_ROW_STYLE,
                                            children=[
                                                html.Button(
                                                    "Emission Totals",
                                                    id="export-emission-totals-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            children=html.Div(
                                                id="export-emission-totals-status",
                                                style={
                                                    "marginTop": "8px",
                                                    "fontSize": "11px",
                                                    "color": COLOR_TEXT_MUTED,
                                                },
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=CARD_STYLE,
                                    children=[
                                        html.Div("Compliance Reports (PDF)", style=EXPORT_SECTION_TITLE_STYLE),
                                        html.Div(
                                            "Generate PDF summaries for reporting windows.",
                                            style=EXPORT_SECTION_HELP_STYLE,
                                        ),
                                        html.Div(
                                            style=EXPORT_BUTTON_ROW_STYLE,
                                            children=[
                                                html.Button(
                                                    "Today",
                                                    id="report-today-btn",
                                                    n_clicks=0,
                                                    style=REPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "This Week",
                                                    id="report-week-btn",
                                                    n_clicks=0,
                                                    style=REPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "This Month",
                                                    id="report-month-btn",
                                                    n_clicks=0,
                                                    style=REPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Previous Month",
                                                    id="report-prev-month-btn",
                                                    n_clicks=0,
                                                    style=REPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "All Time",
                                                    id="report-all-time-btn",
                                                    n_clicks=0,
                                                    style=REPORT_BUTTON_STYLE,
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            children=html.Div(
                                                id="export-report-status",
                                                style={
                                                    "marginTop": "8px",
                                                    "fontSize": "11px",
                                                    "color": COLOR_TEXT_MUTED,
                                                },
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=CARD_STYLE,
                                    children=[
                                        html.Div(
                                            "Exceedances & Failures (PDF)",
                                            style=EXPORT_SECTION_TITLE_STYLE,
                                        ),
                                        html.Div(
                                            "Export incident-focused reports for each time window.",
                                            style=EXPORT_SECTION_HELP_STYLE,
                                        ),
                                        html.Div(
                                            style=EXPORT_BUTTON_ROW_STYLE,
                                            children=[
                                                html.Button(
                                                    "Today",
                                                    id="incident-report-today-btn",
                                                    n_clicks=0,
                                                    style=INCIDENT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "This Week",
                                                    id="incident-report-week-btn",
                                                    n_clicks=0,
                                                    style=INCIDENT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "This Month",
                                                    id="incident-report-month-btn",
                                                    n_clicks=0,
                                                    style=INCIDENT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Previous Month",
                                                    id="incident-report-prev-month-btn",
                                                    n_clicks=0,
                                                    style=INCIDENT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "All Time",
                                                    id="incident-report-all-time-btn",
                                                    n_clicks=0,
                                                    style=INCIDENT_BUTTON_STYLE,
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            children=html.Div(
                                                id="export-incident-status",
                                                style={
                                                    "marginTop": "8px",
                                                    "fontSize": "11px",
                                                    "color": COLOR_TEXT_MUTED,
                                                },
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=CARD_STYLE,
                                    children=[
                                        html.Div(
                                            "Daily Operations Report (PDF)",
                                            style=EXPORT_SECTION_TITLE_STYLE,
                                        ),
                                        html.Div(
                                            "Hourly average charts, CEMS data availability, "
                                            "and operational summary with peak values and weight totals.",
                                            style=EXPORT_SECTION_HELP_STYLE,
                                        ),
                                        html.Div(
                                            style=EXPORT_BUTTON_ROW_STYLE,
                                            children=[
                                                html.Button(
                                                    "Today",
                                                    id="daily-ops-today-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "This Week",
                                                    id="daily-ops-week-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "This Month",
                                                    id="daily-ops-month-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "Previous Month",
                                                    id="daily-ops-prev-month-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                                html.Button(
                                                    "All Time",
                                                    id="daily-ops-all-time-btn",
                                                    n_clicks=0,
                                                    style=EXPORT_BUTTON_STYLE,
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            children=html.Div(
                                                id="export-daily-ops-status",
                                                style={
                                                    "marginTop": "8px",
                                                    "fontSize": "11px",
                                                    "color": COLOR_TEXT_MUTED,
                                                },
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================================
#  CALLBACKS
# ============================================================================


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
    Output("export-minute-download", "data"),
    Output("export-hourly-download", "data"),
    Output("export-rolling-download", "data"),
    Output("export-data-status", "children"),
    Input("export-minute-btn", "n_clicks"),
    Input("export-hourly-btn", "n_clicks"),
    Input("export-rolling-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_data_exports(_minute_clicks, _hourly_clicks, _rolling_clicks):
    trigger = ctx.triggered_id
    if trigger == "export-minute-btn":
        return (
            build_export_payload(MINUTE_CSV, MINUTE_AVG_HEADERS, "minute_averages.csv"),
            no_update,
            no_update,
            "Minute Averages export requested. Preparing your download now.",
        )
    if trigger == "export-hourly-btn":
        return (
            no_update,
            build_export_payload(HOURLY_CSV, HOURLY_AVG_HEADERS, "hourly_averages.csv"),
            no_update,
            "Hourly Averages export requested. Preparing your download now.",
        )
    if trigger == "export-rolling-btn":
        return (
            no_update,
            no_update,
            build_export_payload(
                ROLLING_12HR_CSV,
                ROLLING_AVG_HEADERS,
                "rolling_12hr_averages.csv",
            ),
            "Rolling 12 Hour export requested. Preparing your download now.",
        )
    return no_update, no_update, no_update, no_update


@app.callback(
    Output("export-range-raw-download", "data"),
    Output("export-range-minute-download", "data"),
    Output("export-range-hourly-download", "data"),
    Output("export-range-rolling-download", "data"),
    Output("export-range-status", "children"),
    Input("export-range-raw-btn", "n_clicks"),
    Input("export-range-minute-btn", "n_clicks"),
    Input("export-range-hourly-btn", "n_clicks"),
    Input("export-range-rolling-btn", "n_clicks"),
    State("export-range-dropdown", "value"),
    prevent_initial_call=True,
)
def export_time_range_data_exports(
    _raw_clicks, _minute_clicks, _hourly_clicks, _rolling_clicks, range_key
):
    selected_range = range_key or "month"
    trigger = ctx.triggered_id
    if trigger == "export-range-raw-btn":
        return (
            build_raw_time_range_export_payload(selected_range),
            no_update,
            no_update,
            no_update,
            f"Raw data export requested for {selected_range}. Preparing your download now.",
        )
    if trigger == "export-range-minute-btn":
        return (
            no_update,
            build_time_range_export_payload(
                MINUTE_CSV,
                MINUTE_AVG_HEADERS,
                "minute_averages.csv",
                "minute_end",
                selected_range,
            ),
            no_update,
            no_update,
            f"Minute averages export requested for {selected_range}. Preparing your download now.",
        )
    if trigger == "export-range-hourly-btn":
        return (
            no_update,
            no_update,
            build_time_range_export_payload(
                HOURLY_CSV,
                HOURLY_AVG_HEADERS,
                "hourly_averages.csv",
                "hour_end",
                selected_range,
            ),
            no_update,
            f"Hourly averages export requested for {selected_range}. Preparing your download now.",
        )
    if trigger == "export-range-rolling-btn":
        return (
            no_update,
            no_update,
            no_update,
            build_time_range_export_payload(
                ROLLING_12HR_CSV,
                ROLLING_AVG_HEADERS,
                "rolling_12hr_averages.csv",
                "window_end",
                selected_range,
            ),
            f"Rolling 12 hour export requested for {selected_range}. Preparing your download now.",
        )
    return no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("export-nox-calc-raw-download", "data"),
    Output("export-nox-calc-minute-download", "data"),
    Output("export-nox-calc-hourly-download", "data"),
    Output("export-nox-calc-status", "children"),
    Input("export-nox-calc-raw-btn", "n_clicks"),
    Input("export-nox-calc-minute-btn", "n_clicks"),
    Input("export-nox-calc-hourly-btn", "n_clicks"),
    State("export-range-dropdown", "value"),
    prevent_initial_call=True,
)
def export_nox_manual_calc(_raw_clicks, _minute_clicks, _hourly_clicks, range_key):
    selected_range = range_key or "month"
    trigger = ctx.triggered_id
    if trigger == "export-nox-calc-raw-btn":
        payload = build_nox_manual_calc_payload("raw", selected_range)
        if payload is None:
            return (
                no_update,
                no_update,
                no_update,
                f"No NOx samples found for {selected_range}, or NOx/flow tags are not configured.",
            )
        return (
            payload,
            no_update,
            no_update,
            f"NOx manual-calc (raw) export requested for {selected_range}. Preparing your download now.",
        )
    if trigger == "export-nox-calc-minute-btn":
        payload = build_nox_manual_calc_payload("minute", selected_range)
        if payload is None:
            return (
                no_update,
                no_update,
                no_update,
                f"No NOx minute averages found for {selected_range}, or NOx/flow tags are not configured.",
            )
        return (
            no_update,
            payload,
            no_update,
            f"NOx manual-calc (minute) export requested for {selected_range}. Preparing your download now.",
        )
    if trigger == "export-nox-calc-hourly-btn":
        payload = build_nox_manual_calc_payload("hourly", selected_range)
        if payload is None:
            return (
                no_update,
                no_update,
                no_update,
                f"No NOx hourly averages found for {selected_range}, or NOx/flow tags are not configured.",
            )
        return (
            no_update,
            no_update,
            payload,
            f"NOx manual-calc (hourly) export requested for {selected_range}. Preparing your download now.",
        )
    return no_update, no_update, no_update, no_update


@app.callback(
    Output("export-emission-totals-download", "data"),
    Output("export-emission-totals-status", "children"),
    Input("export-emission-totals-btn", "n_clicks"),
    State("export-range-dropdown", "value"),
    prevent_initial_call=True,
)
def export_emission_totals(_clicks, range_key):
    selected_range = range_key or "month"
    payload = build_emission_totals_payload(selected_range)
    if payload is None:
        return (
            no_update,
            f"No emission data found for {selected_range}, or NOx/CO/flow tags are not configured.",
        )
    return (
        payload,
        f"Emission totals export requested for {selected_range} (with YTD and trailing 12 months). Preparing your download now.",
    )


@app.callback(
    Output("report-today-download", "data"),
    Output("report-week-download", "data"),
    Output("report-month-download", "data"),
    Output("report-prev-month-download", "data"),
    Output("report-all-time-download", "data"),
    Output("export-report-status", "children"),
    Input("report-today-btn", "n_clicks"),
    Input("report-week-btn", "n_clicks"),
    Input("report-month-btn", "n_clicks"),
    Input("report-prev-month-btn", "n_clicks"),
    Input("report-all-time-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_compliance_reports(
    _today_clicks, _week_clicks, _month_clicks, _prev_month_clicks, _all_time_clicks
):
    trigger = ctx.triggered_id
    if trigger == "report-today-btn":
        return (
            build_report_payload("today"),
            no_update,
            no_update,
            no_update,
            no_update,
            "Today's compliance report requested. Generating your PDF now.",
        )
    if trigger == "report-week-btn":
        return (
            no_update,
            build_report_payload("week"),
            no_update,
            no_update,
            no_update,
            "This Week compliance report requested. Generating your PDF now.",
        )
    if trigger == "report-month-btn":
        return (
            no_update,
            no_update,
            build_report_payload("month"),
            no_update,
            no_update,
            "This Month compliance report requested. Generating your PDF now.",
        )
    if trigger == "report-prev-month-btn":
        return (
            no_update,
            no_update,
            no_update,
            build_report_payload("prev_month"),
            no_update,
            "Previous Month compliance report requested. Generating your PDF now.",
        )
    if trigger == "report-all-time-btn":
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            build_report_payload("all_time"),
            "All Time compliance report requested. Generating your PDF now.",
        )
    return no_update, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("incident-report-today-download", "data"),
    Output("incident-report-week-download", "data"),
    Output("incident-report-month-download", "data"),
    Output("incident-report-prev-month-download", "data"),
    Output("incident-report-all-time-download", "data"),
    Output("export-incident-status", "children"),
    Input("incident-report-today-btn", "n_clicks"),
    Input("incident-report-week-btn", "n_clicks"),
    Input("incident-report-month-btn", "n_clicks"),
    Input("incident-report-prev-month-btn", "n_clicks"),
    Input("incident-report-all-time-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_incident_reports(
    _today_clicks, _week_clicks, _month_clicks, _prev_month_clicks, _all_time_clicks
):
    trigger = ctx.triggered_id
    if trigger == "incident-report-today-btn":
        return (
            build_incident_report_payload("today"),
            no_update,
            no_update,
            no_update,
            no_update,
            "Today's incident report requested. Generating your PDF now.",
        )
    if trigger == "incident-report-week-btn":
        return (
            no_update,
            build_incident_report_payload("week"),
            no_update,
            no_update,
            no_update,
            "This Week incident report requested. Generating your PDF now.",
        )
    if trigger == "incident-report-month-btn":
        return (
            no_update,
            no_update,
            build_incident_report_payload("month"),
            no_update,
            no_update,
            "This Month incident report requested. Generating your PDF now.",
        )
    if trigger == "incident-report-prev-month-btn":
        return (
            no_update,
            no_update,
            no_update,
            build_incident_report_payload("prev_month"),
            no_update,
            "Previous Month incident report requested. Generating your PDF now.",
        )
    if trigger == "incident-report-all-time-btn":
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            build_incident_report_payload("all_time"),
            "All Time incident report requested. Generating your PDF now.",
        )
    return no_update, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("daily-ops-today-download", "data"),
    Output("daily-ops-week-download", "data"),
    Output("daily-ops-month-download", "data"),
    Output("daily-ops-prev-month-download", "data"),
    Output("daily-ops-all-time-download", "data"),
    Output("export-daily-ops-status", "children"),
    Input("daily-ops-today-btn", "n_clicks"),
    Input("daily-ops-week-btn", "n_clicks"),
    Input("daily-ops-month-btn", "n_clicks"),
    Input("daily-ops-prev-month-btn", "n_clicks"),
    Input("daily-ops-all-time-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_daily_ops_reports(
    _today_clicks, _week_clicks, _month_clicks, _prev_month_clicks, _all_time_clicks
):
    trigger = ctx.triggered_id
    range_map = {
        "daily-ops-today-btn": ("today", 0),
        "daily-ops-week-btn": ("week", 1),
        "daily-ops-month-btn": ("month", 2),
        "daily-ops-prev-month-btn": ("prev_month", 3),
        "daily-ops-all-time-btn": ("all_time", 4),
    }
    entry = range_map.get(trigger)
    if not entry:
        return no_update, no_update, no_update, no_update, no_update, no_update
    range_key, idx = entry
    result = [no_update] * 5
    result[idx] = build_daily_ops_payload(range_key)
    label = range_key.replace("_", " ").title()
    result.append(f"{label} daily operations report requested. Generating your PDF now.")
    return tuple(result)


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
        rolling_df = load_rolling_12hr_stats()
        live_rolling_df = load_live_rolling_12hr_stats()
        rolling_combined_df = merge_live_rolling_12hr(rolling_df, live_rolling_df)
        thresholds = load_thresholds()
        config_version = compute_config_version_text()
        alias_map = build_alias_lookup(
            raw_df,
            hourly_df,
            rolling_combined_df,
            load_alias_map_from_settings(),
        )

        _, _, last_ts = extract_raw_stats(raw_df)
        last_full_hour_stats = extract_last_full_hour(hourly_df)
        rolling_12hr_stats = extract_latest_rolling_12hr(rolling_combined_df)

        epa_settings = load_epa_settings()
        current_hour_lb_hr = {
            tag: avg_lb_hr for tag, (_, avg_lb_hr, _, _, _) in last_full_hour_stats.items()
        }
        current_hour_avg = {
            tag: avg for tag, (avg, _, _, _, _) in last_full_hour_stats.items()
        }

        tags = sorted(
            set(current_hour_lb_hr.keys())
            | set(rolling_12hr_stats.keys())
            | set(load_configured_tags_from_settings())
            | set(thresholds.keys()),
            key=str,
        )
        tags = [
            t for t in tags
            if not looks_numeric_tag(str(t)) and not str(t).startswith("EPA19:")
        ]

        cems_map = {
            "o2": "CEMS O2",
            "nox": "CEMS NOX",
            "co": "CEMS CO",
        }

        def _has_lbhr_data(tag_name: Optional[str]) -> bool:
            if not tag_name:
                return False
            current_val = current_hour_lb_hr.get(tag_name, float("nan"))
            rolling_val = rolling_12hr_stats.get(tag_name, (float("nan"), float("nan"), None, None, 0))[1]
            return (current_val == current_val) or (rolling_val == rolling_val)

        cards = []
        flow_tag = str(epa_settings.get("epa_flow_tag", "") or "")
        if not flow_tag:
            flow_tag = find_flow_tag(tags, thresholds, alias_map)
        cards.append(
            build_flow_card(
                label="Air Flow",
                tag=flow_tag or None,
                current_hour_avg=current_hour_avg,
                rolling_12hr_stats=rolling_12hr_stats,
                thresholds=thresholds,
                alias_map=alias_map,
            )
        )

        # Processing Time gauge
        machine_state_tag = load_machine_state_tag_from_settings()
        processing_is_running = False
        if machine_state_tag:
            current_hr_proc, today_proc = compute_processing_time_minutes(
                raw_df, machine_state_tag
            )
            all_time_proc = compute_all_time_processing_minutes(machine_state_tag)
            cards.append(
                build_processing_time_card(current_hr_proc, today_proc, all_time_proc)
            )
            processing_is_running = (
                isinstance(current_hr_proc, (int, float))
                and current_hr_proc == current_hr_proc
                and current_hr_proc > 0
            )

        # Feed System / Conveyor status card
        feed_settings = load_feed_settings()
        if feed_settings.get("detection", "none") != "none":
            feed_status = compute_feed_status(raw_df, feed_settings)
            cards.append(
                build_feed_status_card(feed_status, processing_is_running)
            )


        for metric, label in cems_map.items():
            ppm_tag = find_cems_tag(metric, tags, thresholds, alias_map)
            if ppm_tag:
                primary_units = "%" if metric == "o2" else "ppm"
                cards.append(
                    build_cems_card(
                        label=f"{label} ({primary_units})",
                        tag=ppm_tag,
                        current_hour_lb_hr=current_hour_lb_hr,
                        current_hour_avg=current_hour_avg,
                        rolling_12hr_stats=rolling_12hr_stats,
                        thresholds=thresholds,
                        alias_map=alias_map,
                        units_label=primary_units,
                        value_source="avg_value",
                    )
                )

            # O2 is displayed as a single percent gauge only.
            if metric == "o2":
                continue

            # CO/NOx lb/hr is taken from the same pollutant tag's calculated avg_lb_hr.
            if ppm_tag and _has_lbhr_data(ppm_tag):
                # Inject default permit limit for lb/hr gauge when no
                # high_limit is set in thresholds.json
                lbhr_thresholds = dict(thresholds)
                default_limit = DEFAULT_PERMIT_LIMITS_LB_HR.get(metric)
                if default_limit is not None:
                    existing = dict(lbhr_thresholds.get(ppm_tag, {}) or {})
                    if existing.get("high_limit") is None:
                        existing["high_limit"] = default_limit
                        lbhr_thresholds[ppm_tag] = existing

                cards.append(
                    build_cems_card(
                        label=f"{label} (lb/hr)",
                        tag=ppm_tag,
                        current_hour_lb_hr=current_hour_lb_hr,
                        current_hour_avg=current_hour_avg,
                        rolling_12hr_stats=rolling_12hr_stats,
                        thresholds=lbhr_thresholds,
                        alias_map=alias_map,
                        units_label="lb/hr",
                        value_source="avg_lb_hr",
                    )
                )

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
            style={**CARD_STYLE, "borderTop": f"3px solid {COLOR_BAD}"},
            children=[
                html.Div(
                    "Dashboard Error",
                    style={**CARD_HEADER_STYLE, "color": COLOR_BAD},
                ),
                html.Div(
                    f"An error occurred while updating the dashboard: {e}",
                    style={"fontSize": "12px", "color": COLOR_TEXT_MUTED},
                ),
            ],
        )
        last_update = "Last update: error"
        return [error_card], last_update, compute_config_version_text()


# ============================================================================
#  MAIN ENTRY POINT
# ============================================================================

# ============================================================================
#  HEADLESS CLI -- reports, exports, readable exports, and self-verification
#
#  Usage (no args launches the dashboard as before):
#    python CIPMonitor.py report   --type compliance|incident|ops|all --range RANGE|all [--out DIR]
#    python CIPMonitor.py export    --kind raw|hourly|minute|rolling     --range RANGE|all [--out DIR]
#    python CIPMonitor.py readable  --granularity raw|hourly             --range RANGE|all [--format csv|xlsx] [--out DIR]
#    python CIPMonitor.py metrics   --range RANGE|all [--out DIR]
#    python CIPMonitor.py verify    --range RANGE|all [--tolerance 0.01]
#    python CIPMonitor.py all       --range RANGE|all [--out DIR]
#  RANGE in: today week month prev_month all_time
# ============================================================================

CLI_RANGES = ["today", "week", "month", "prev_month", "all_time"]


def _report_window(range_key: str) -> Tuple[str, datetime, datetime]:
    """Resolve a report range exactly as the PDF generators do."""
    data_start = _get_report_data_start(load_rolling_12hr_stats(), load_hourly_stats())
    return _get_report_range(range_key, data_start=data_start)


def compute_report_metrics(range_key: str) -> Dict[str, object]:
    """Authoritative numeric metrics for a report range.

    Uses the SAME primitives and tag-discovery the PDF generators use, so the
    CLI ``metrics`` dump and ``verify`` checks reflect exactly what the reports
    print. Returns a JSON-serializable dict (datetimes -> ISO strings; no sets).
    """
    rolling_df = load_rolling_12hr_stats()
    hourly_df = load_hourly_stats()
    data_start = _get_report_data_start(rolling_df, hourly_df)
    label, start, end = _get_report_range(range_key, data_start=data_start)
    alias_map = build_alias_lookup(
        pd.DataFrame(), hourly_df, rolling_df, load_alias_map_from_settings()
    )
    thresholds = load_thresholds()
    epa_settings = load_epa_settings()

    rolling_range = _filter_time_range(rolling_df, "window_end", start, end)
    hourly_range = _filter_time_range(hourly_df, "hour_start", start, end)

    co_pattern = [re.compile(r"\bco\b|co[_\s-]", re.IGNORECASE)]
    nox_pattern = [re.compile(r"\bnox\b|no[_\s-]?x", re.IGNORECASE)]
    o2_pattern = [re.compile(r"\bo2\b|o2[_\s-]", re.IGNORECASE)]
    co_tag = (_find_tag_by_patterns(rolling_range, alias_map, co_pattern)
              or _find_tag_by_patterns(hourly_range, alias_map, co_pattern))
    nox_tag = (_find_tag_by_patterns(rolling_range, alias_map, nox_pattern)
               or _find_tag_by_patterns(hourly_range, alias_map, nox_pattern))
    o2_tag = (_find_tag_by_patterns(rolling_range, alias_map, o2_pattern)
              or _find_tag_by_patterns(hourly_range, alias_map, o2_pattern))

    tags = sorted(set(hourly_range.get("tag", pd.Series(dtype=str)).dropna().astype(str)))
    flow_tag = str(epa_settings.get("epa_flow_tag", "") or "") or find_flow_tag(tags, thresholds, alias_map)
    machine_state_tag = load_machine_state_tag_from_settings()

    days_span = max(1, (datetime.now().date() - start.date()).days + 2)
    raw_history = load_raw_history(max_days=min(days_span, 400))
    processing_stats = compute_processing_time_range(raw_history, machine_state_tag, start, end)
    processing_hour_starts = processing_stats.get("hour_starts_with_any", set())

    feed_settings = load_feed_settings()
    weight_tag = feed_settings.get("weight_tag", "")
    weight_run = compute_weight_decrease_time_range(raw_history, weight_tag, start, end) if weight_tag else None

    co_weight = _compute_total_weight(hourly_range, co_tag, flow_tag, EPA19_CO_MOLECULAR_WEIGHT, processing_hour_starts)
    nox_weight = _compute_total_weight(hourly_range, nox_tag, flow_tag, EPA19_NOX_MOLECULAR_WEIGHT, processing_hour_starts)
    flow_avg = _compute_avg_flow(hourly_range, flow_tag)

    cems_tags = [t for t in [co_tag, nox_tag, o2_tag] if t]
    uptime = compute_cems_uptime(hourly_df, machine_state_tag, cems_tags, start, end)
    total_report_hours = float(uptime.get("total_hours", 0)) or ((end - start).total_seconds() / 3600.0)

    # Incident metrics
    exceedances_df = load_exceedances()
    env_events_df = load_env_events()
    exceed_range = _filter_time_range(exceedances_df, "start_time", start, end)
    exceed_count = int(exceed_range.shape[0]) if not exceed_range.empty else 0
    exceed_minutes = (
        float(exceed_range["duration_sec"].fillna(0).astype(float).sum()) / 60.0
        if (not exceed_range.empty and "duration_sec" in exceed_range.columns) else 0.0
    )
    system_events = _filter_time_range(env_events_df, "timestamp", start, end)
    if not system_events.empty and "event_type" in system_events.columns:
        system_events = system_events[system_events["event_type"].astype(str).apply(_is_system_failure)]
    failure_count = int(system_events.shape[0]) if not system_events.empty else 0
    failure_minutes = (
        float(system_events["duration_sec"].fillna(0).astype(float).sum()) / 60.0
        if (not system_events.empty and "duration_sec" in system_events.columns) else 0.0
    )
    threshold_summary = compute_compliance_summary(load_latest_raw_df(), thresholds, exceedances_df)
    system_health = load_system_health()

    def _f(x):
        try:
            xf = float(x)
            return xf if xf == xf else None
        except Exception:
            return None

    return {
        "range_key": range_key,
        "label": label,
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "window_hours": round((end - start).total_seconds() / 3600.0, 3),
        "tags": {
            "co": co_tag, "nox": nox_tag, "o2": o2_tag,
            "flow": flow_tag, "machine_state": machine_state_tag,
        },
        "rows": {
            "hourly_in_range": int(len(hourly_range)),
            "rolling_in_range": int(len(rolling_range)),
            "raw_history_rows": int(len(raw_history)) if raw_history is not None else 0,
        },
        "processing": {
            "total_hours": _f(processing_stats.get("total_hours")),
            "total_minutes": _f(processing_stats.get("total_minutes")),
            "capped_gap_count": int(processing_stats.get("capped_gap_count", 0)),
            "sample_count": int(processing_stats.get("sample_count", 0)),
            "hours_with_any": int(len(processing_hour_starts)),
        },
        "flow_avg": _f(flow_avg),
        "co_total_weight_lb": _f(co_weight),
        "nox_total_weight_lb": _f(nox_weight),
        "weight_run": None if not weight_run else {
            "total_hours": _f(weight_run.get("total_hours")),
            "avg_feed_rate_per_min": _f(weight_run.get("avg_feed_rate_per_min")),
            "sample_count": int(weight_run.get("sample_count", 0)),
        },
        "uptime": {
            "uptime_pct": _f(uptime.get("uptime_pct")),
            "operating_hours": int(uptime.get("operating_hours", 0)),
            "valid_cems_hours": int(uptime.get("valid_cems_hours", 0)),
            "missing_hours": int(uptime.get("missing_hours", 0)),
            "total_hours": _f(uptime.get("total_hours")),
            "report_avail_pct": round(
                (int(uptime.get("valid_cems_hours", 0)) / total_report_hours * 100.0), 1
            ) if total_report_hours > 0 else 0.0,
        },
        "incident": {
            "exceedance_count": exceed_count,
            "exceedance_minutes": round(exceed_minutes, 2),
            "system_failure_count": failure_count,
            "system_failure_minutes": round(failure_minutes, 2),
            "pct_within_limits_24h": _f(threshold_summary.get("pct_24h")),
            "pct_within_limits_30d": _f(threshold_summary.get("pct_30d")),
            "health_status": str(system_health.get("status", "Unknown")),
        },
        "molar_volume_dscf": EPA19_MOLAR_VOLUME_DSCF,
    }


def _load_raw_range_df(range_key: str) -> pd.DataFrame:
    """Concatenated raw samples filtered to a report range [start, end)."""
    _, start, end = _report_window(range_key)
    paths = _get_raw_daily_paths_for_range(start, end)
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            for col in RAW_DATA_HEADERS:
                if col not in df.columns:
                    df[col] = None
            frames.append(df.reindex(columns=RAW_DATA_HEADERS))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=RAW_DATA_HEADERS)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return _filter_time_range(df, "timestamp", start, end)


def _alias_for(tag: str, alias_map: Dict[str, str]) -> str:
    a = alias_map.get(tag) if alias_map else None
    return str(a).strip() if a else str(tag)


def cli_export_long(kind: str, range_key: str, out_dir: str) -> Optional[str]:
    """Export raw/hourly/minute/rolling data for a range in the native long format."""
    ensure_dir(out_dir)
    _, start, end = _report_window(range_key)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_dir, f"{range_key}_{kind}_{stamp}.csv")
    if kind == "raw":
        df = _load_raw_range_df(range_key)
    elif kind == "hourly":
        df = _filter_time_range(load_hourly_stats(), "hour_start", start, end)
    elif kind == "minute":
        path = os.path.join(LOG_DIR, "minute_averages.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if "minute_start" in df.columns:
            df["minute_start"] = pd.to_datetime(df["minute_start"], errors="coerce")
            df = _filter_time_range(df, "minute_start", start, end)
    elif kind == "rolling":
        df = _filter_time_range(load_rolling_12hr_stats(), "window_end", start, end)
    else:
        return None
    if df is None or df.empty:
        # still write a header-only file so the export is "available"
        pd.DataFrame().to_csv(out, index=False)
        return out
    df.to_csv(out, index=False)
    return out


def cli_export_readable(granularity: str, range_key: str, fmt: str, out_dir: str) -> Optional[str]:
    """Human-readable export: one row per timestamp, one column per alias.

    raw     -> wide table of raw sample values (last value per timestamp/alias)
    hourly  -> wide table of hourly averages, plus per-hour CO/NOx lb/hr columns
    """
    ensure_dir(out_dir)
    _, start, end = _report_window(range_key)
    alias_map = build_alias_lookup(
        pd.DataFrame(), load_hourly_stats(), load_rolling_12hr_stats(),
        load_alias_map_from_settings(),
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if granularity == "raw":
        df = _load_raw_range_df(range_key)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["timestamp"])
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
        if "qa_flag" in df.columns:
            df = df[df["qa_flag"].astype(str).str.upper().isin({"OK", "OUT_OF_RANGE", "MANUAL_CORRECTION"})]
        df["col"] = df["tag"].map(lambda t: _alias_for(str(t), alias_map))
        wide = df.pivot_table(index="timestamp", columns="col", values="value_num", aggfunc="last")
        wide = wide.sort_index()
        wide.index.name = "Timestamp"
        title = f"Raw samples (wide) — {range_key}"
    elif granularity == "hourly":
        hr = _filter_time_range(load_hourly_stats(), "hour_start", start, end)
        if hr is None or hr.empty:
            return None
        hr = hr.copy()
        hr["col"] = hr["tag"].map(lambda t: _alias_for(str(t), alias_map))
        wide = hr.pivot_table(index="hour_start", columns="col", values="avg_value", aggfunc="last")
        wide = wide.sort_index()
        # add per-hour CO / NOx lb/hr
        epa = load_epa_settings()
        flow_tag = str(epa.get("epa_flow_tag", "") or "") or find_flow_tag(
            sorted(set(hr["tag"].dropna().astype(str))), load_thresholds(), alias_map)
        for ptag, mw, name in (
            (str(epa.get("epa_co_tag", "") or ""), EPA19_CO_MOLECULAR_WEIGHT, "CO lb/hr"),
            (str(epa.get("epa_nox_tag", "") or ""), EPA19_NOX_MOLECULAR_WEIGHT, "NOx lb/hr"),
        ):
            rates = _hourly_lb_hr_recomputed(hr, ptag or None, flow_tag or None, mw)
            if not rates.empty:
                s = rates.set_index("hour_start")["lb_hr"]
                wide[name] = s.reindex(wide.index)
        wide.index.name = "Hour"
        title = f"Hourly averages (wide) — {range_key}"
    else:
        return None

    wide = wide.round(4)
    if fmt == "xlsx":
        out = os.path.join(out_dir, f"{range_key}_{granularity}_readable_{stamp}.xlsx")
        metrics = compute_report_metrics(range_key)
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            summary = pd.DataFrame(
                [(k, v) for k, v in _flatten_metrics(metrics).items()],
                columns=["Metric", "Value"],
            )
            summary.to_excel(xw, sheet_name="Summary", index=False)
            wide.to_excel(xw, sheet_name="Data")
            for sh in xw.sheets.values():
                sh.freeze_panes = "A2"
        return out
    out = os.path.join(out_dir, f"{range_key}_{granularity}_readable_{stamp}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {title}\n")
        wide.to_csv(f)
    return out


def _flatten_metrics(m: Dict[str, object], prefix: str = "") -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for k, v in m.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat


def cli_generate_report(rtype: str, range_key: str, out_dir: str) -> Optional[str]:
    """Generate a real PDF report (same generators the dashboard buttons use)."""
    ensure_dir(out_dir)
    gen = {
        "compliance": generate_report_pdf,
        "incident": generate_incident_report_pdf,
        "ops": generate_daily_ops_pdf,
    }.get(rtype)
    if gen is None:
        return None
    path = gen(range_key)
    if not path or not os.path.exists(path):
        return None
    dest = os.path.join(out_dir, os.path.basename(path))
    if os.path.abspath(dest) != os.path.abspath(path):
        shutil.copyfile(path, dest)
    return dest


# ---------------------------------------------------------------------------
#  Performance-test run blocks (arbitrary time windows) + CO ppmvd @ 7% O2
# ---------------------------------------------------------------------------

def _rnd(x, n: int = 4):
    """Round to n places, mapping NaN/invalid to None (JSON/CSV friendly)."""
    if x is None:
        return None
    try:
        xf = float(x)
        return round(xf, n) if xf == xf else None
    except (TypeError, ValueError):
        return None


def _correct_to_ref_o2(conc, o2_pct, ref_o2: float = 7.0, ambient_o2: float = 20.9):
    """Correct a dry-basis concentration to a reference O2 %.

    C_ref = C_meas * (ambient_o2 - ref_o2) / (ambient_o2 - O2_meas)

    This is the standard EPA diluent correction (40 CFR 60 App. A / Subpart EEE
    uses 7% O2 for the CO/HC standards). Requires the concentration and O2 to be
    on a DRY basis. Returns None if O2 is missing or >= ambient (no valid
    combustion dilution / division by zero).
    """
    try:
        c = float(conc)
        o = float(o2_pct)
    except (TypeError, ValueError):
        return None
    if c != c or o != o:  # NaN guard
        return None
    denom = ambient_o2 - o
    if denom <= 0:
        return None
    return c * (ambient_o2 - ref_o2) / denom


def _ppm_flow_to_lbhr(ppm, flow, mw):
    """EPA Method 19 mass rate: ppm * flow_dscfm * 60 * MW / (1e6 * Vm)."""
    if ppm is None or flow is None:
        return None
    try:
        return float(ppm) * float(flow) * 60.0 * float(mw) / (1_000_000.0 * EPA19_MOLAR_VOLUME_DSCF)
    except (TypeError, ValueError):
        return None


def _load_raw_between(start: datetime, end: datetime) -> pd.DataFrame:
    """Concatenated raw samples in [start, end), across the daily files it spans.

    Mirrors _load_raw_range_df but for arbitrary (start, end) rather than a named
    range. Handles both raw_data_*.csv and .csv.gz (pandas infers compression).
    """
    paths = _get_raw_daily_paths_for_range(start, end)
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            for col in RAW_DATA_HEADERS:
                if col not in df.columns:
                    df[col] = None
            frames.append(df.reindex(columns=RAW_DATA_HEADERS))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=RAW_DATA_HEADERS)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return _filter_time_range(df, "timestamp", start, end)


def _parse_run_windows(file_path: Optional[str], windows_str: Optional[str]) -> List[Tuple[str, datetime, datetime]]:
    """Parse run windows from a CSV file and/or an inline string.

    CSV file: needs 'start' and 'stop' (or 'end') columns; optional 'run'/'label'.
    Inline:   'start,stop[,label]; start,stop; ...' (datetimes have no commas).
    Datetimes are parsed permissively (ISO, 'YYYY-MM-DD HH:MM', 'M/D/YYYY HH:MM').
    """
    rows: List[Tuple[str, datetime, datetime]] = []
    if file_path:
        wdf = pd.read_csv(file_path)
        cols = {str(c).lower().strip(): c for c in wdf.columns}
        scol = cols.get("start") or cols.get("start_time") or cols.get("begin")
        ecol = cols.get("stop") or cols.get("end") or cols.get("end_time") or cols.get("finish")
        lcol = cols.get("run") or cols.get("label") or cols.get("name")
        if not scol or not ecol:
            raise ValueError("windows file needs 'start' and 'stop' (or 'end') columns")
        for i, r in wdf.iterrows():
            s = pd.to_datetime(r[scol], errors="coerce")
            e = pd.to_datetime(r[ecol], errors="coerce")
            if pd.isna(s) or pd.isna(e):
                continue
            lbl = str(r[lcol]) if (lcol and pd.notna(r.get(lcol))) else f"run{len(rows) + 1}"
            rows.append((lbl, s.to_pydatetime(), e.to_pydatetime()))
    if windows_str:
        for part in (p for p in windows_str.split(";") if p.strip()):
            bits = [b.strip() for b in part.split(",")]
            if len(bits) < 2:
                continue
            s = pd.to_datetime(bits[0], errors="coerce")
            e = pd.to_datetime(bits[1], errors="coerce")
            if pd.isna(s) or pd.isna(e):
                continue
            lbl = bits[2] if len(bits) >= 3 and bits[2] else f"run{len(rows) + 1}"
            rows.append((str(lbl), s.to_pydatetime(), e.to_pydatetime()))
    return rows


RUN_BLOCK_COLUMNS = [
    "run", "start", "stop", "duration_min",
    "O2_pct", "CO_ppmvd_raw", "CO_ppmvd_7pctO2", "CO_ppmvd_7pctO2_runavg",
    "NOx_ppm", "CO_lbhr", "NOx_lbhr", "flow_dscfm",
    "CO_samples", "minutes_with_data", "capture_pct",
]


def compute_run_blocks(
    windows: List[Tuple[str, datetime, datetime]],
    ref_o2: float = 7.0,
    ambient_o2: float = 20.9,
    co_already_corrected: bool = False,
) -> pd.DataFrame:
    """Average each run window and derive the compliance-relevant metrics.

    Per window [start, stop): block-average O2, CO, NOx, and flow over all valid
    samples (qa_flag in OK/OUT_OF_RANGE/MANUAL_CORRECTION); derive CO/NOx lb/hr by
    EPA Method 19 from the block averages; and compute CO ppmvd @ ref O2 two ways:
      - CO_ppmvd_7pctO2:        correct each 1-minute CO value by that minute's O2,
                                then average the corrected minutes (EEE/CEMS method).
      - CO_ppmvd_7pctO2_runavg: correct the run-average CO by the run-average O2.
    If co_already_corrected is set, the CO channel is treated as already at ref O2
    and the corrected columns equal the measured CO.
    """
    epa = load_epa_settings()
    o2_tag = str(epa.get("epa_o2_tag", "") or "")
    co_tag = str(epa.get("epa_co_tag", "") or "")
    nox_tag = str(epa.get("epa_nox_tag", "") or "")
    flow_tag = str(epa.get("epa_flow_tag", "") or "")
    accept = {"OK", "OUT_OF_RANGE", "MANUAL_CORRECTION"}

    out_rows: List[Dict[str, object]] = []
    for (label, start, stop) in windows:
        dur_min = (stop - start).total_seconds() / 60.0
        base: Dict[str, object] = {
            "run": label,
            "start": start.isoformat(sep=" ", timespec="seconds"),
            "stop": stop.isoformat(sep=" ", timespec="seconds"),
            "duration_min": round(dur_min, 3),
        }
        raw = _load_raw_between(start, stop)
        if raw is None or raw.empty:
            out_rows.append({**base, **{k: None for k in RUN_BLOCK_COLUMNS if k not in base},
                             "CO_samples": 0, "minutes_with_data": 0, "capture_pct": 0.0})
            continue

        r = raw.copy()
        if "qa_flag" in r.columns:
            r = r[r["qa_flag"].astype(str).str.upper().isin(accept)]
        r["v"] = pd.to_numeric(r["value"], errors="coerce")
        r = r.dropna(subset=["v", "timestamp"])
        if r.empty:
            out_rows.append({**base, **{k: None for k in RUN_BLOCK_COLUMNS if k not in base},
                             "CO_samples": 0, "minutes_with_data": 0, "capture_pct": 0.0})
            continue
        r["minute"] = r["timestamp"].dt.floor("min")

        def tag_mean(tag: str):
            if not tag:
                return None
            vv = r.loc[r["tag"] == tag, "v"]
            return float(vv.mean()) if not vv.empty else None

        def tag_minute_means(tag: str) -> pd.Series:
            if not tag:
                return pd.Series(dtype=float)
            return r.loc[r["tag"] == tag].groupby("minute")["v"].mean()

        o2_avg = tag_mean(o2_tag)
        co_avg = tag_mean(co_tag)
        nox_avg = tag_mean(nox_tag)
        flow_avg = tag_mean(flow_tag)

        # CO @ ref O2 — minute-corrected then averaged (EEE-consistent)
        co7_minute = None
        if co_already_corrected:
            co7_minute = co_avg
        else:
            co_min = tag_minute_means(co_tag)
            o2_min = tag_minute_means(o2_tag)
            if not co_min.empty and not o2_min.empty:
                joined = pd.DataFrame({"co": co_min, "o2": o2_min}).dropna()
                if not joined.empty:
                    corr = [
                        _correct_to_ref_o2(c, o, ref_o2, ambient_o2)
                        for c, o in zip(joined["co"], joined["o2"])
                    ]
                    corr = [c for c in corr if c is not None]
                    if corr:
                        co7_minute = sum(corr) / len(corr)

        # CO @ ref O2 — run-average corrected (secondary, for comparison)
        co7_runavg = co_avg if co_already_corrected else _correct_to_ref_o2(co_avg, o2_avg, ref_o2, ambient_o2)

        # The 7% O2 correction is only meaningful under combustion conditions
        # (run-average O2 below ambient). If the window's average O2 is at/above
        # ambient there is no net dilution, and the per-minute correction becomes
        # numerically unstable near the 1/(ambient - O2) singularity -- so suppress
        # both corrected values rather than emit noise-amplified numbers. Real test
        # runs (O2 well below ambient) are unaffected.
        if not co_already_corrected and (o2_avg is None or o2_avg >= ambient_o2):
            co7_minute = None
            co7_runavg = None

        co_lbhr = _ppm_flow_to_lbhr(co_avg, flow_avg, EPA19_CO_MOLECULAR_WEIGHT)
        nox_lbhr = _ppm_flow_to_lbhr(nox_avg, flow_avg, EPA19_NOX_MOLECULAR_WEIGHT)

        co_samples = int((r["tag"] == co_tag).sum()) if co_tag else 0
        minutes_with_data = int(tag_minute_means(co_tag).shape[0]) if co_tag else 0
        capture = round(minutes_with_data / dur_min * 100.0, 1) if dur_min > 0 else None

        out_rows.append({
            **base,
            "O2_pct": _rnd(o2_avg),
            "CO_ppmvd_raw": _rnd(co_avg),
            "CO_ppmvd_7pctO2": _rnd(co7_minute),
            "CO_ppmvd_7pctO2_runavg": _rnd(co7_runavg),
            "NOx_ppm": _rnd(nox_avg),
            "CO_lbhr": _rnd(co_lbhr),
            "NOx_lbhr": _rnd(nox_lbhr),
            "flow_dscfm": _rnd(flow_avg),
            "CO_samples": co_samples,
            "minutes_with_data": minutes_with_data,
            "capture_pct": capture,
        })

    return pd.DataFrame(out_rows, columns=RUN_BLOCK_COLUMNS)


def cli_run_blocks(
    file_path: Optional[str],
    windows_str: Optional[str],
    ref_o2: float,
    ambient_o2: float,
    co_already_corrected: bool,
    fmt: str,
    out_dir: str,
) -> Optional[List[str]]:
    """Compute per-run averages for arbitrary test windows and write CSV/XLSX."""
    ensure_dir(out_dir)
    windows = _parse_run_windows(file_path, windows_str)
    if not windows:
        print("[runblocks] no valid windows (use --file <csv> or --windows 'start,stop; ...')")
        return None
    df = compute_run_blocks(windows, ref_o2, ambient_o2, co_already_corrected)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "ref_o2_pct": ref_o2,
        "ambient_o2_pct": ambient_o2,
        "co_basis_assumed": "dry",
        "co_already_corrected": co_already_corrected,
        "co_7pctO2_method": "correct each 1-minute value by that minute's O2, then average over the run",
        "correction_formula": f"C_ref = C * ({ambient_o2} - {ref_o2}) / ({ambient_o2} - O2)",
        "lbhr_formula": f"ppm * flow_dscfm * 60 * MW / (1e6 * {EPA19_MOLAR_VOLUME_DSCF})",
        "nox_mw": EPA19_NOX_MOLECULAR_WEIGHT,
        "co_mw": EPA19_CO_MOLECULAR_WEIGHT,
        "window_convention": "[start, stop) half-open",
        "o2_tag": str(load_epa_settings().get("epa_o2_tag", "")),
        "co_tag": str(load_epa_settings().get("epa_co_tag", "")),
        "nox_tag": str(load_epa_settings().get("epa_nox_tag", "")),
        "flow_tag": str(load_epa_settings().get("epa_flow_tag", "")),
    }
    outputs: List[str] = []
    if fmt in ("csv", "both"):
        out = os.path.join(out_dir, f"runblocks_{stamp}.csv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write("# SN-31B performance-test run blocks\n")
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
            df.to_csv(f, index=False)
        outputs.append(out)
    if fmt in ("xlsx", "both"):
        out = os.path.join(out_dir, f"runblocks_{stamp}.xlsx")
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="Runs", index=False)
            mdf = pd.DataFrame([(k, str(v)) for k, v in meta.items()], columns=["Field", "Value"])
            mdf.to_excel(xw, sheet_name="Meta", index=False)
            for sh in xw.sheets.values():
                sh.freeze_panes = "A2"
        outputs.append(out)
    for o in outputs:
        print(f"[runblocks] OK -> {o}")
    print(df.to_string(index=False))
    return outputs


def _verify_range(range_key: str, tolerance: float = 0.01) -> Dict[str, object]:
    """Independently recompute key metrics straight from the CSVs and compare to
    compute_report_metrics(). Returns {checks: [...], passed: bool}.

    The recompute here deliberately does NOT reuse the app's aggregation
    helpers (only the raw/hourly CSVs), so it is a genuine cross-check of the
    primitives, not a tautology.
    """
    m = compute_report_metrics(range_key)
    _, start, end = _report_window(range_key)
    checks: List[Dict[str, object]] = []

    def cmp(name, expected, actual, tol=tolerance, abstol=0.05):
        ok = False
        if expected is None and actual is None:
            ok = True
        elif expected is not None and actual is not None:
            denom = max(1e-9, abs(expected))
            ok = abs(expected - actual) <= max(abstol, tol * denom)
        checks.append({
            "check": name, "report": expected, "independent": actual,
            "pass": bool(ok),
        })

    mst = m["tags"]["machine_state"]
    raw = _load_raw_range_df(range_key)

    # --- processing minutes (state 3, OK, gaps clipped [0,120]) ---
    proc_min = 0.0
    proc_hours_set = set()
    if mst and raw is not None and not raw.empty:
        s = raw[raw["tag"] == mst].copy()
        if "qa_flag" in s.columns:
            s = s[s["qa_flag"].astype(str).str.upper() == "OK"]
        s["v"] = pd.to_numeric(s["value"], errors="coerce")
        s = s.dropna(subset=["v", "timestamp"]).sort_values("timestamp")
        s["dt"] = s["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0, upper=120)
        proc = (s["v"] >= 2.5) & (s["v"] <= 3.5)
        proc_min = float(s.loc[proc, "dt"].sum()) / 60.0
        prows = s.loc[proc & (s["dt"] > 0)]
        proc_hours_set = set(prows["timestamp"].dt.floor("h"))
    cmp("processing_total_minutes", m["processing"]["total_minutes"], round(proc_min, 2))
    cmp("processing_hours_with_any", m["processing"]["hours_with_any"], len(proc_hours_set), abstol=0.5)

    # --- flow avg (mean of hourly avg_value for flow tag over hour_start in range) ---
    hr = _filter_time_range(load_hourly_stats(), "hour_start", start, end)
    flow_tag = m["tags"]["flow"]
    flow_avg_ind = None
    if flow_tag and hr is not None and not hr.empty:
        fv = pd.to_numeric(hr.loc[hr["tag"] == flow_tag, "avg_value"], errors="coerce").dropna()
        if not fv.empty:
            flow_avg_ind = float(fv.mean())
    cmp("flow_avg", m["flow_avg"], flow_avg_ind)

    # --- CO / NOx total weight (sum ppm*flow*60*MW/(1e6*Vm) over processing hours) ---
    def weight_independent(ptag, mw):
        if not ptag or not flow_tag or hr is None or hr.empty:
            return None
        p = hr.loc[hr["tag"] == ptag, ["hour_start", "avg_value"]].rename(columns={"avg_value": "ppm"})
        fl = hr.loc[hr["tag"] == flow_tag, ["hour_start", "avg_value"]].rename(columns={"avg_value": "flow"})
        j = p.merge(fl, on="hour_start", how="inner")
        j["ppm"] = pd.to_numeric(j["ppm"], errors="coerce")
        j["flow"] = pd.to_numeric(j["flow"], errors="coerce")
        j = j.dropna(subset=["ppm", "flow"])
        if proc_hours_set:
            j = j[j["hour_start"].isin(proc_hours_set)]
        if j.empty:
            return None
        return float((j["ppm"] * j["flow"] * 60.0 * mw / (1_000_000.0 * EPA19_MOLAR_VOLUME_DSCF)).sum())
    cmp("co_total_weight_lb", m["co_total_weight_lb"], weight_independent(m["tags"]["co"], EPA19_CO_MOLECULAR_WEIGHT))
    cmp("nox_total_weight_lb", m["nox_total_weight_lb"], weight_independent(m["tags"]["nox"], EPA19_NOX_MOLECULAR_WEIGHT))

    # --- operating hours (hourly state avg in [2.5,3.5]) ---
    op_ind = None
    if mst and hr is not None and not hr.empty:
        sv = pd.to_numeric(hr.loc[hr["tag"] == mst, "avg_value"], errors="coerce")
        op_ind = int(((sv >= 2.5) & (sv <= 3.5)).sum())
    cmp("operating_hours", m["uptime"]["operating_hours"], op_ind, abstol=0.5)

    passed = all(c["pass"] for c in checks)
    return {"range_key": range_key, "passed": passed, "checks": checks, "metrics": m}


def _cli_main(argv: List[str]) -> int:
    import argparse, json
    parser = argparse.ArgumentParser(prog="CIPMonitor", description="Headless CIP reports & exports")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_range(sp):
        sp.add_argument("--range", default="today", help="range or 'all' (" + " ".join(CLI_RANGES) + ")")
        sp.add_argument("--out", default=os.path.join(EXPORT_TMP_DIR, "cli"), help="output directory")

    p_rep = sub.add_parser("report", help="generate PDF report(s)")
    p_rep.add_argument("--type", default="all", choices=["compliance", "incident", "ops", "all"])
    add_range(p_rep)

    p_exp = sub.add_parser("export", help="export native long-format CSV")
    p_exp.add_argument("--kind", default="raw", choices=["raw", "hourly", "minute", "rolling"])
    add_range(p_exp)

    p_read = sub.add_parser("readable", help="human-readable wide export")
    p_read.add_argument("--granularity", default="hourly", choices=["raw", "hourly"])
    p_read.add_argument("--format", default="xlsx", choices=["csv", "xlsx"])
    add_range(p_read)

    p_met = sub.add_parser("metrics", help="dump report metrics as JSON")
    add_range(p_met)

    p_ver = sub.add_parser("verify", help="independently recompute & compare metrics")
    p_ver.add_argument("--tolerance", type=float, default=0.01)
    add_range(p_ver)

    p_all = sub.add_parser("all", help="reports + exports + readable + metrics + verify")
    add_range(p_all)

    p_rb = sub.add_parser("runblocks", help="per-run averages for arbitrary test windows (incl. CO ppmvd @ 7% O2)")
    p_rb.add_argument("--file", default=None, help="CSV of windows with start,stop[,run] columns")
    p_rb.add_argument("--windows", default=None, help="inline 'start,stop[,label]; start,stop; ...'")
    p_rb.add_argument("--ref-o2", type=float, default=7.0, help="reference O2 %% for correction (EEE = 7)")
    p_rb.add_argument("--ambient-o2", type=float, default=20.9, help="ambient O2 %% basis (default 20.9)")
    p_rb.add_argument("--co-already-corrected", action="store_true",
                      help="set if the CO channel is already corrected to ref O2 at the analyzer")
    p_rb.add_argument("--format", default="both", choices=["csv", "xlsx", "both"])
    p_rb.add_argument("--out", default=os.path.join(EXPORT_TMP_DIR, "cli"), help="output directory")

    args = parser.parse_args(argv)

    if args.cmd == "runblocks":
        try:
            res = cli_run_blocks(
                args.file, args.windows, args.ref_o2, args.ambient_o2,
                args.co_already_corrected, args.format, args.out,
            )
            return 0 if res else 1
        except Exception as e:
            print(f"[runblocks] ERROR {e}")
            return 1

    ranges = CLI_RANGES if getattr(args, "range", "today") == "all" else [args.range]
    out_dir = getattr(args, "out", os.path.join(EXPORT_TMP_DIR, "cli"))
    ensure_dir(out_dir)
    rc = 0

    for rk in ranges:
        if rk not in CLI_RANGES:
            print(f"[skip] unknown range: {rk}")
            continue
        if args.cmd in ("report", "all"):
            types = ["compliance", "incident", "ops"] if (args.cmd == "all" or args.type == "all") else [args.type]
            for t in types:
                try:
                    p = cli_generate_report(t, rk, out_dir)
                    print(f"[report:{t}:{rk}] {'OK -> ' + p if p else 'FAILED'}")
                except Exception as e:
                    print(f"[report:{t}:{rk}] ERROR {e}"); rc = 1
        if args.cmd in ("export", "all"):
            kinds = ["raw", "hourly", "minute", "rolling"] if args.cmd == "all" else [args.kind]
            for k in kinds:
                try:
                    p = cli_export_long(k, rk, out_dir)
                    print(f"[export:{k}:{rk}] {'OK -> ' + p if p else 'no data'}")
                except Exception as e:
                    print(f"[export:{k}:{rk}] ERROR {e}"); rc = 1
        if args.cmd in ("readable", "all"):
            grans = ["raw", "hourly"] if args.cmd == "all" else [args.granularity]
            fmt = "xlsx" if args.cmd == "all" else args.format
            for g in grans:
                try:
                    p = cli_export_readable(g, rk, fmt, out_dir)
                    print(f"[readable:{g}:{rk}] {'OK -> ' + p if p else 'no data'}")
                except Exception as e:
                    print(f"[readable:{g}:{rk}] ERROR {e}"); rc = 1
        if args.cmd in ("metrics", "all"):
            try:
                m = compute_report_metrics(rk)
                mp = os.path.join(out_dir, f"{rk}_metrics.json")
                with open(mp, "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=2, default=str)
                print(f"[metrics:{rk}] OK -> {mp}")
                print(json.dumps(m, indent=2, default=str))
            except Exception as e:
                print(f"[metrics:{rk}] ERROR {e}"); rc = 1
        if args.cmd in ("verify", "all"):
            try:
                res = _verify_range(rk, getattr(args, "tolerance", 0.01))
                status = "PASS" if res["passed"] else "FAIL"
                print(f"[verify:{rk}] {status}")
                for c in res["checks"]:
                    flag = "ok " if c["pass"] else "XX "
                    print(f"    {flag}{c['check']:<28} report={c['report']} independent={c['independent']}")
                if not res["passed"]:
                    rc = 2
            except Exception as e:
                print(f"[verify:{rk}] ERROR {e}"); rc = 1
    return rc


# ============================================================================
#  MAIN ENTRY POINT
# ============================================================================

_CLI_COMMANDS = {"report", "export", "readable", "metrics", "verify", "all", "runblocks"}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_COMMANDS:
        raise SystemExit(_cli_main(sys.argv[1:]))
    app.run(host="0.0.0.0", port=8050, debug=False)
