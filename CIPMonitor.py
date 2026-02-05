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
        "pandas is required for CIPMonitor. Install via 'pip install pandas'."
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

# ----------------- config -----------------

LOG_DIR = "logs"
MINUTE_CSV = os.path.join(LOG_DIR, "minute_averages.csv")
HOURLY_CSV = os.path.join(LOG_DIR, "hourly_averages.csv")
ROLLING_12HR_CSV = os.path.join(LOG_DIR, "rolling_12hr_averages.csv")
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

REFRESH_MS = 5000  # dashboard refresh interval (ms)

EPA19_STD_O2_PCT = 20.9
EPA19_MOLAR_VOLUME_SCF = 385.8
EPA19_MOLECULAR_WEIGHTS = {
    "NOx": 46.0,  # as NO2
    "CO": 28.01,
    "O2": 32.0,
}

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
    mask = (df[time_col] >= start) & (df[time_col] <= end)
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


def _compute_peak(
    df: pd.DataFrame, tag: str, value_col: str, time_col: str
) -> Tuple[Optional[float], Optional[datetime]]:
    if not tag or df is None or df.empty:
        return None, None
    subset = df.loc[df["tag"] == tag].dropna(subset=[value_col, time_col])
    if subset.empty:
        return None, None
    try:
        idx = subset[value_col].idxmax()
    except Exception:
        return None, None
    row = subset.loc[idx]
    return _to_float(row.get(value_col)), row.get(time_col)


def _compute_running_hours(
    hourly_df: pd.DataFrame,
    tags: List[str],
    start: datetime,
    end: datetime,
) -> Tuple[int, int, int]:
    if hourly_df is None or hourly_df.empty or not tags:
        return 0, 0, 0
    hourly_range = _filter_time_range(hourly_df, "hour_end", start, end)
    if hourly_range.empty:
        return 0, 0, 0
    hourly_range = hourly_range.loc[hourly_range["tag"].isin(tags)]
    if hourly_range.empty:
        return 0, 0, 0
    total_hours = 0
    running_hours = 0
    for hour_end, group in hourly_range.groupby("hour_end"):
        if pd.isna(hour_end):
            continue
        total_hours += 1
        values = group.get("avg_lb_hr")
        if values is None:
            values = group.get("avg_value")
        if values is None:
            values = group["avg_value"] if "avg_value" in group else None
        if values is not None and (pd.to_numeric(values, errors="coerce").fillna(0) > 0).any():
            running_hours += 1
    not_running = max(total_hours - running_hours, 0)
    return total_hours, running_hours, not_running


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


def _compute_total_weight(hourly_range: pd.DataFrame, tag: Optional[str]) -> Optional[float]:
    if not tag or hourly_range is None or hourly_range.empty:
        return None
    tag_series = hourly_range.loc[hourly_range["tag"] == tag]
    if tag_series.empty:
        return None
    values = tag_series.get("avg_lb_hr")
    if values is None:
        values = tag_series.get("avg_value")
    if values is None:
        return None
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


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
    hourly_range = _filter_time_range(hourly_df, "hour_end", start, end)

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

    co_series = rolling_range.loc[rolling_range["tag"] == co_tag] if co_tag else pd.DataFrame()
    nox_series = rolling_range.loc[rolling_range["tag"] == nox_tag] if nox_tag else pd.DataFrame()
    o2_series = rolling_range.loc[rolling_range["tag"] == o2_tag] if o2_tag else pd.DataFrame()

    co_entry = thresholds.get(co_tag, {}) if co_tag and isinstance(thresholds.get(co_tag, {}), dict) else {}
    nox_entry = thresholds.get(nox_tag, {}) if nox_tag and isinstance(thresholds.get(nox_tag, {}), dict) else {}
    o2_entry = thresholds.get(o2_tag, {}) if o2_tag and isinstance(thresholds.get(o2_tag, {}), dict) else {}

    co_high_oper = _to_float(co_entry.get("high_oper"))
    co_high_limit = _to_float(co_entry.get("high_limit"))
    nox_high_oper = _to_float(nox_entry.get("high_oper"))
    nox_high_limit = _to_float(nox_entry.get("high_limit"))
    o2_high_oper = _to_float(o2_entry.get("high_oper"))
    o2_high_limit = _to_float(o2_entry.get("high_limit"))

    co_peak, co_peak_time = _compute_peak(rolling_range, co_tag, "avg_lb_hr", "window_end")
    nox_peak, nox_peak_time = _compute_peak(
        rolling_range, nox_tag, "avg_lb_hr", "window_end"
    )
    o2_peak, o2_peak_time = _compute_peak(rolling_range, o2_tag, "avg_value", "window_end")

    total_hours, running_hours, not_running_hours = _compute_running_hours(
        hourly_range, [tag for tag in [co_tag, nox_tag] if tag], start, end
    )

    tags = sorted(set(hourly_range.get("tag", pd.Series(dtype=str)).dropna().astype(str)))
    flow_tag = str(epa_settings.get("epa_flow_tag", "") or "")
    if not flow_tag:
        flow_tag = find_flow_tag(tags, thresholds, alias_map)
    flow_avg = _compute_avg_flow(hourly_range, flow_tag)
    flow_label = _display_name(flow_tag, alias_map, "Flow")
    co_weight = _compute_total_weight(hourly_range, co_tag)
    nox_weight = _compute_total_weight(hourly_range, nox_tag)

    report_text_primary = "#1b1f2a"
    report_text_muted = "#3b4a5f"
    report_text_subtle = "#5a6b82"
    report_bg = "#f4f7fb"
    report_panel = "#ffffff"
    report_grid = "#d6e0ef"
    report_table_header = "#dbe9f6"
    report_table_alt = "#f1f6fc"
    report_border = "#b7c7dd"
    report_line_primary = "#1f77b4"
    report_line_secondary = "#ff7f0e"
    report_line_tertiary = "#2ca02c"
    report_limit_line = "#6c6c6c"

    ensure_dir(EXPORT_TMP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cip_report_{range_key}_{timestamp}.pdf"
    report_path = os.path.join(EXPORT_TMP_DIR, filename)

    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(report_bg)
        gs = fig.add_gridspec(3, 1, height_ratios=[0.18, 0.55, 0.27])
        fig.subplots_adjust(right=0.78)

        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.axis("off")
        title_text = f"CIP Emissions Report - {label}"
        subtitle = f"Reporting Window: {_format_report_dt(start)} to {_format_report_dt(end)}"
        ax_title.text(
            0.0,
            0.75,
            title_text,
            fontsize=18,
            fontweight="bold",
            color=report_text_primary,
        )
        ax_title.text(
            0.0,
            0.4,
            subtitle,
            fontsize=11,
            color=report_text_muted,
        )
        ax_title.text(
            0.0,
            0.1,
            "Rolling 12-hour averages for CO/NOx (lb/hr) with %O2 trend overlay.",
            fontsize=9.5,
            color=report_text_subtle,
        )

        ax_chart = fig.add_subplot(gs[1, 0])
        ax_chart.set_title("Rolling 12-Hour Averages", fontsize=12, color=report_text_primary)
        ax_chart.set_ylabel("lb/hr", color=report_text_primary)
        ax_chart.set_facecolor(report_panel)
        ax_chart.grid(True, linestyle="--", color=report_grid, alpha=0.6)
        ax_chart.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d\n%H:%M"))

        lines = []
        labels = []

        if not co_series.empty:
            line, = ax_chart.plot(
                co_series["window_end"],
                co_series["avg_lb_hr"],
                color=report_line_primary,
                linewidth=2,
                linestyle="-",
            )
            lines.append(line)
            labels.append(f"CO 12-hr avg ({_display_name(co_tag, alias_map, 'CO')})")

        if not nox_series.empty:
            line, = ax_chart.plot(
                nox_series["window_end"],
                nox_series["avg_lb_hr"],
                color=report_line_secondary,
                linewidth=2,
                linestyle="--",
            )
            lines.append(line)
            labels.append(f"NOx 12-hr avg ({_display_name(nox_tag, alias_map, 'NOx')})")

        ax_o2 = ax_chart.twinx()
        ax_o2.set_ylabel("%O2", color=report_text_primary)
        ax_o2.set_facecolor("none")
        if not o2_series.empty:
            line, = ax_o2.plot(
                o2_series["window_end"],
                o2_series["avg_value"],
                color=report_line_tertiary,
                linestyle=":",
                linewidth=2,
            )
            lines.append(line)
            labels.append(f"O2 % ({_display_name(o2_tag, alias_map, 'O2')})")

        co_limit_value = co_high_oper
        nox_limit_value = nox_high_oper
        o2_limit_value = None

        if co_limit_value is not None:
            _ensure_limit_visible(ax_chart, co_limit_value)
            _append_limit_lines(
                ax_chart,
                "CO",
                co_high_oper,
                None,
                lines,
                labels,
                color=report_limit_line,
            )
        if nox_limit_value is not None:
            _ensure_limit_visible(ax_chart, nox_limit_value)
            _append_limit_lines(
                ax_chart,
                "NOx",
                nox_high_oper,
                None,
                lines,
                labels,
                color=report_limit_line,
            )

        if lines:
            ax_chart.legend(
                lines,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
                frameon=False,
                borderaxespad=0.0,
            )
        else:
            ax_chart.text(
                0.5,
                0.5,
                "No rolling 12-hour data available for the selected window.",
                ha="center",
                va="center",
                fontsize=11,
                color=report_text_muted,
                transform=ax_chart.transAxes,
            )

        ax_table = fig.add_subplot(gs[2, 0])
        ax_table.axis("off")

        summary_rows = [
            ["CO Peak (lb/hr)", f"{co_peak:.2f} @ {_format_report_dt(co_peak_time)}" if co_peak is not None else "N/A"],
            ["NOx Peak (lb/hr)", f"{nox_peak:.2f} @ {_format_report_dt(nox_peak_time)}" if nox_peak is not None else "N/A"],
            ["O2 Peak (%)", f"{o2_peak:.2f} @ {_format_report_dt(o2_peak_time)}" if o2_peak is not None else "N/A"],
            [
                f"Average Flow Rate ({flow_label})",
                f"{flow_avg:.2f}" if flow_avg is not None else "N/A",
            ],
            [
                "CO Total Weight (lb)",
                f"{co_weight:.2f}" if co_weight is not None else "N/A",
            ],
            [
                "NOx Total Weight (lb)",
                f"{nox_weight:.2f}" if nox_weight is not None else "N/A",
            ],
            ["Total Hours Tracked", f"{total_hours}"],
            ["Running Hours", f"{running_hours}"],
            ["Not Running Hours", f"{not_running_hours}"],
        ]

        table = ax_table.table(
            cellText=summary_rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.4)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color=report_text_primary)
                cell.set_facecolor(report_table_header)
            else:
                cell.set_facecolor(report_table_alt)
                cell.set_edgecolor(report_border)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return report_path


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

    report_text_primary = "#1b1f2a"
    report_text_muted = "#3b4a5f"
    report_text_subtle = "#5a6b82"
    report_bg = "#f4f7fb"
    report_table_header = "#dbe9f6"
    report_table_alt = "#f1f6fc"
    report_border = "#b7c7dd"

    ensure_dir(EXPORT_TMP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cip_incident_report_{range_key}_{timestamp}.pdf"
    report_path = os.path.join(EXPORT_TMP_DIR, filename)

    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(report_bg)
        gs = fig.add_gridspec(3, 1, height_ratios=[0.2, 0.4, 0.4])

        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.axis("off")
        title_text = f"CIP Exceedances & System Failures - {label}"
        subtitle = f"Reporting Window: {_format_report_dt(start)} to {_format_report_dt(end)}"
        ax_title.text(
            0.0,
            0.75,
            title_text,
            fontsize=17,
            fontweight="bold",
            color=report_text_primary,
        )
        ax_title.text(
            0.0,
            0.4,
            subtitle,
            fontsize=11,
            color=report_text_muted,
        )
        ax_title.text(
            0.0,
            0.1,
            "Exceedances reflect regulatory limit violations; system failures summarize health and data events.",
            fontsize=9.5,
            color=report_text_subtle,
        )

        ax_summary = fig.add_subplot(gs[1, 0])
        ax_summary.axis("off")
        summary_rows = [
            ["Exceedance events", str(exceed_count)],
            ["Exceedance duration", f"{exceed_minutes:.1f} min"],
            ["System failure events", str(failure_count)],
            ["Failure duration", f"{failure_minutes:.1f} min"],
            ["Within limits (last 24h)", f"{threshold_summary.get('pct_24h', 0.0):.1f}%"],
            ["Within limits (last 30d)", f"{threshold_summary.get('pct_30d', 0.0):.1f}%"],
            ["System health status", str(system_health.get("status", "Unknown"))],
            ["System health detail", str(system_health.get("status_reason", ""))],
        ]
        summary_table = ax_summary.table(
            cellText=summary_rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(10)
        summary_table.scale(1, 1.3)
        for (row, col), cell in summary_table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color=report_text_primary)
                cell.set_facecolor(report_table_header)
            else:
                cell.set_facecolor(report_table_alt)
                cell.set_edgecolor(report_border)

        ax_tables = fig.add_subplot(gs[2, 0])
        ax_tables.axis("off")

        def _prepare_exceed_rows(df: pd.DataFrame) -> List[List[str]]:
            if df.empty:
                return [["No exceedance events in this window.", "", "", ""]]
            rows: List[List[str]] = []
            for _, row in df.sort_values("start_time", ascending=False).head(8).iterrows():
                rows.append(
                    [
                        str(row.get("tag", "")),
                        _format_report_dt(pd.to_datetime(row.get("start_time"), errors="coerce")),
                        _format_report_dt(pd.to_datetime(row.get("end_time"), errors="coerce")),
                        _format_duration(row.get("duration_sec")),
                    ]
                )
            return rows

        def _prepare_failure_rows(df: pd.DataFrame) -> List[List[str]]:
            if df.empty:
                return [["No system failure events in this window.", "", "", ""]]
            rows: List[List[str]] = []
            for _, row in df.sort_values("timestamp", ascending=False).head(8).iterrows():
                rows.append(
                    [
                        str(row.get("event_type", "")),
                        str(row.get("tag", "")),
                        _format_report_dt(pd.to_datetime(row.get("timestamp"), errors="coerce")),
                        _format_duration(row.get("duration_sec")),
                    ]
                )
            return rows

        exceed_rows = _prepare_exceed_rows(exceed_range)
        failure_rows = _prepare_failure_rows(system_events)

        table_ax = ax_tables.inset_axes([0.0, 0.05, 0.48, 0.9])
        table_ax.axis("off")
        table_ax.set_title("Recent Exceedances", fontsize=11, color=report_text_primary, pad=6)
        exceed_table = table_ax.table(
            cellText=exceed_rows,
            colLabels=["Tag", "Start", "End", "Duration"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        exceed_table.auto_set_font_size(False)
        exceed_table.set_fontsize(9)
        exceed_table.scale(1, 1.2)

        failure_ax = ax_tables.inset_axes([0.52, 0.05, 0.48, 0.9])
        failure_ax.axis("off")
        failure_ax.set_title(
            "System Failures / Errors",
            fontsize=11,
            color=report_text_primary,
            pad=6,
        )
        failure_table = failure_ax.table(
            cellText=failure_rows,
            colLabels=["Type", "Tag", "Timestamp", "Duration"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        failure_table.auto_set_font_size(False)
        failure_table.set_fontsize(9)
        failure_table.scale(1, 1.2)

        for table in (exceed_table, failure_table):
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight="bold", color=report_text_primary)
                    cell.set_facecolor(report_table_header)
                else:
                    cell.set_facecolor(report_table_alt)
                    cell.set_edgecolor(report_border)

        fig.tight_layout()
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
    candidates: List[Tuple[datetime, str]] = []

    for name in os.listdir(LOG_DIR):
        if not (name.startswith("raw_data_") and name.endswith(".csv")):
            continue
        try:
            date_part = name.replace("raw_data_", "").replace(".csv", "")
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except Exception:
            continue
        if file_date < cutoff_date:
            continue
        candidates.append((datetime.combine(file_date, datetime.min.time()), os.path.join(LOG_DIR, name)))

    if not candidates:
        return pd.DataFrame(columns=base_cols)

    candidates.sort(key=lambda tup: tup[0])

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


def load_epa_ppm_to_lbhr_map() -> Dict[str, str]:
    if not os.path.exists(SETTINGS_JSON):
        return {}
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    mapping: Dict[str, str] = {}
    tags_text = data.get("tags", "")
    _, alias_map = parse_tags_and_aliases(tags_text)
    nox_tag = resolve_tag_from_alias(str(data.get("epa_nox_tag", "") or ""), alias_map)
    co_tag = resolve_tag_from_alias(str(data.get("epa_co_tag", "") or ""), alias_map)
    o2_tag = resolve_tag_from_alias(str(data.get("epa_o2_tag", "") or ""), alias_map)

    if nox_tag:
        mapping[nox_tag] = "EPA19:NOx_LBHR"
    if co_tag:
        mapping[co_tag] = "EPA19:CO_LBHR"
    if o2_tag:
        mapping[o2_tag] = "EPA19:O2_LBHR"
    return mapping


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


def _epa_o2_values(raw_o2: Optional[float], units: str) -> Tuple[Optional[float], Optional[float]]:
    if raw_o2 is None:
        return None, None
    if units == "ppmv":
        o2_ppmv = raw_o2
        o2_pct = raw_o2 / 10000.0
    else:
        o2_pct = raw_o2
        o2_ppmv = raw_o2 * 10000.0
    return o2_pct, o2_ppmv


def _correct_ppmv_for_o2(ppmv: float, o2_pct: float, ref_o2_pct: float) -> Optional[float]:
    if not (0.0 <= o2_pct < EPA19_STD_O2_PCT):
        return None
    if not (0.0 <= ref_o2_pct < EPA19_STD_O2_PCT):
        return None
    return ppmv * (EPA19_STD_O2_PCT - ref_o2_pct) / (EPA19_STD_O2_PCT - o2_pct)


def compute_rolling_lbhr_from_epa(
    tag: str,
    rolling_12hr_stats: Dict[str, Tuple[float, float, datetime, datetime, int]],
    epa_settings: Dict[str, object],
) -> Optional[float]:
    if not epa_settings or not epa_settings.get("epa_enabled"):
        return None

    flow_tag = str(epa_settings.get("epa_flow_tag", "") or "")
    o2_tag = str(epa_settings.get("epa_o2_tag", "") or "")
    o2_units = str(epa_settings.get("epa_o2_units", "percent") or "percent")
    ref_o2_pct = float(epa_settings.get("epa_ref_o2_pct", 3.0) or 3.0)

    pollutant_map = {
        str(epa_settings.get("epa_nox_tag", "") or ""): "NOx",
        str(epa_settings.get("epa_co_tag", "") or ""): "CO",
        str(epa_settings.get("epa_o2_tag", "") or ""): "O2",
    }

    pollutant = pollutant_map.get(tag)
    if not pollutant:
        return None

    flow_entry = rolling_12hr_stats.get(flow_tag)
    tag_entry = rolling_12hr_stats.get(tag)
    if not flow_entry or not tag_entry:
        return None

    flow_avg = flow_entry[0]
    tag_avg = tag_entry[0]
    if flow_avg != flow_avg or tag_avg != tag_avg:
        return None

    o2_entry = rolling_12hr_stats.get(o2_tag)
    o2_avg = o2_entry[0] if o2_entry else None
    o2_pct, o2_ppmv = _epa_o2_values(o2_avg, o2_units)

    if pollutant == "O2":
        ppmv = o2_ppmv
    else:
        if o2_pct is None:
            return None
        corrected = _correct_ppmv_for_o2(tag_avg, o2_pct, ref_o2_pct)
        if corrected is None:
            return None
        ppmv = corrected

    if ppmv is None:
        return None

    mw = EPA19_MOLECULAR_WEIGHTS.get(pollutant)
    if mw is None:
        return None

    return (ppmv * flow_avg * 60.0 * mw) / (1_000_000.0 * EPA19_MOLAR_VOLUME_SCF)


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
    return COLOR_TEXT_SUBTLE


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

    window_df = window_df[window_df.get("ok", False)]
    total = 0
    within = 0

    for _, row in window_df.iterrows():
        tag = str(row.get("tag"))
        entry = thresholds.get(tag, {}) if isinstance(thresholds.get(tag, {}), dict) else {}
        low_limit = _to_float(entry.get("low_limit"))
        high_limit = _to_float(entry.get("high_limit"))
        try:
            val = float(row.get("value_num"))
        except Exception:
            continue
        if val != val:
            continue

        total += 1
        if _sample_within_limits(val, low_limit, high_limit):
            within += 1

    if total == 0:
        return 0.0

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

    pct_24h = compute_within_limit_percent(raw_df, thresholds, timedelta(hours=24))
    pct_30d = compute_within_limit_percent(raw_df, thresholds, timedelta(days=30))

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


# ----------------- styling helpers -----------------

COLOR_BG = "#0f141b"
COLOR_SURFACE = "#1b232d"
COLOR_SURFACE_ALT = "#212a36"
COLOR_BORDER = "#2a3442"
COLOR_TEXT_PRIMARY = "#f2f5f9"
COLOR_TEXT_MUTED = "#a3adba"
COLOR_TEXT_SUBTLE = "#c1c9d4"
COLOR_ACCENT = "#7c5cff"
COLOR_ACCENT_BORDER = "#6d4ee6"
COLOR_BUTTON_SECONDARY = "#53657a"
COLOR_BUTTON_SECONDARY_BORDER = "#46576b"
COLOR_BUTTON_TERTIARY = "#617487"
COLOR_BUTTON_TERTIARY_BORDER = "#556679"
COLOR_TABLE_HEADER = "#273240"
COLOR_TABLE_CELL = "#1b232d"
COLOR_TABLE_CELL_ALT = "#202a38"

CARD_STYLE = {
    "backgroundColor": COLOR_SURFACE,
    "borderRadius": "10px",
    "border": f"1px solid {COLOR_BORDER}",
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

EXPORT_SECTION_TITLE_STYLE = {
    "fontSize": "13px",
    "fontWeight": "600",
    "color": COLOR_TEXT_PRIMARY,
}

EXPORT_SECTION_HELP_STYLE = {
    "fontSize": "11px",
    "color": COLOR_TEXT_MUTED,
}

EXPORT_BUTTON_ROW_STYLE = {
    "display": "flex",
    "gap": "10px",
    "flexWrap": "wrap",
}

EXPORT_BUTTON_STYLE = {
    "backgroundColor": COLOR_ACCENT,
    "color": "white",
    "border": f"1px solid {COLOR_ACCENT_BORDER}",
    "padding": "8px 12px",
    "borderRadius": "8px",
    "fontSize": "12px",
    "fontWeight": "600",
}

REPORT_BUTTON_STYLE = {
    "backgroundColor": COLOR_BUTTON_SECONDARY,
    "color": "white",
    "border": f"1px solid {COLOR_BUTTON_SECONDARY_BORDER}",
    "padding": "8px 12px",
    "borderRadius": "8px",
    "fontSize": "12px",
    "fontWeight": "600",
}

INCIDENT_BUTTON_STYLE = {
    "backgroundColor": COLOR_BUTTON_TERTIARY,
    "color": "white",
    "border": f"1px solid {COLOR_BUTTON_TERTIARY_BORDER}",
    "padding": "8px 12px",
    "borderRadius": "8px",
    "fontSize": "12px",
    "fontWeight": "600",
}


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


def build_cems_card(
    label: str,
    tag: Optional[str],
    current_hour_lb_hr: Dict[str, float],
    current_hour_avg: Dict[str, float],
    rolling_12hr_stats: Dict[str, Tuple[float, float, datetime, datetime, int]],
    thresholds: Dict[str, Dict[str, object]],
    alias_map: Optional[Dict[str, str]] = None,
    units_label: str = "lb/hr",
    use_avg_value: bool = False,
) -> html.Div:
    entry = thresholds.get(tag, {}) if tag and isinstance(thresholds.get(tag, {}), dict) else {}
    alias = entry.get("alias") if isinstance(entry.get("alias"), str) else None
    if not alias and tag and alias_map:
        alias = alias_map.get(tag)

    low_oper = entry.get("low_oper")
    high_oper = entry.get("high_oper")
    low_limit = entry.get("low_limit")
    high_limit = entry.get("high_limit")

    if use_avg_value:
        value = current_hour_avg.get(tag, float("nan")) if tag else float("nan")
    else:
        value = current_hour_lb_hr.get(tag, float("nan")) if tag else float("nan")
    rolling_entry = rolling_12hr_stats.get(
        tag, (float("nan"), float("nan"), None, None, 0)
    )
    rolling_avg_value, rolling_avg_lb_hr, ws, we, rolling_count = rolling_entry
    rolling_avg = rolling_avg_value if use_avg_value else rolling_avg_lb_hr

    low_for_range = low_oper if low_oper is not None else low_limit
    high_for_range = high_oper if high_oper is not None else high_limit
    low_eff, high_eff, gmin, gmax = compute_gauge_range(
        low_for_range, high_for_range, [value, rolling_avg]
    )
    low_for_class = low_oper if low_oper is not None else low_eff
    high_for_class = high_oper if high_oper is not None else high_eff

    status = classify_value(value, low_for_class, high_for_class)
    color = status_color(status)

    if value == value:
        value_label = f"{value:.2f}"
    else:
        value_label = "—"

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

    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(
                header,
                style={"fontWeight": "600", "fontSize": "13px", "color": COLOR_TEXT_PRIMARY},
            ),
            html.Div(subtitle, style={"fontSize": "11px", "color": COLOR_TEXT_MUTED}),
            html.Div(
                style=GAUGE_CONTAINER_STYLE,
                children=[
                    daq.Gauge(
                        id={"type": "cems-gauge", "tag": tag or label},
                        min=gmin,
                        max=gmax,
                        value=value if value == value else gmin,
                        showCurrentValue=True,
                        color=color,
                        label=f"Current hourly avg: {value_label}",
                        size=200,
                        units=units_label,
                    ),
                ],
            ),
            html.Div(rolling_text, style={"fontSize": "11px", "color": COLOR_TEXT_MUTED}),
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
        low_for_range, high_for_range, [value, rolling_avg]
    )
    low_for_class = low_oper if low_oper is not None else low_eff
    high_for_class = high_oper if high_oper is not None else high_eff

    status = classify_value(value, low_for_class, high_for_class)
    color = status_color(status)

    if value == value:
        value_label = f"{value:.2f}"
    else:
        value_label = "—"

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

    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(
                header,
                style={"fontWeight": "600", "fontSize": "13px", "color": COLOR_TEXT_PRIMARY},
            ),
            html.Div(subtitle, style={"fontSize": "11px", "color": COLOR_TEXT_MUTED}),
            html.Div(
                style=GAUGE_CONTAINER_STYLE,
                children=[
                    daq.Gauge(
                        id={"type": "flow-gauge", "tag": tag or label},
                        min=gmin,
                        max=gmax,
                        value=value if value == value else gmin,
                        showCurrentValue=True,
                        color=color,
                        label=f"Current hourly avg: {value_label}",
                        size=200,
                        units=units_label,
                    ),
                ],
            ),
            html.Div(rolling_text, style={"fontSize": "11px", "color": COLOR_TEXT_MUTED}),
        ],
    )


def build_system_health_card(health: Dict[str, object]) -> html.Div:
    status = str(health.get("status", "Unknown"))
    status_reason = str(health.get("status_reason", ""))
    status_upper = status.lower()
    if "critical" in status_upper:
        color = "#ef9a9a"
    elif "degraded" in status_upper:
        color = "#ffb74d"
    elif "healthy" in status_upper:
        color = "#a5d6a7"
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
                style={"fontSize": "11px"},
            )
        )

    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(
                "System Health",
                style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "6px"},
            ),
            html.Div(status, style={"color": color, "fontSize": "20px", "fontWeight": "700"}),
            html.Div(
                status_reason,
                style={"fontSize": "11px", "color": COLOR_TEXT_MUTED, "marginBottom": "6px"},
            ),
            html.Div(details, style={"display": "flex", "flexDirection": "column", "gap": "2px"}),
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
            html.Div(
                "Data Quality (last 24h)",
                style={"fontWeight": "600", "fontSize": "13px"},
            ),
            html.Div(
                "QA flags drive valid %; DATA_GAP events summarize missing runs.",
                style={"fontSize": "11px", "color": COLOR_TEXT_MUTED},
            ),
            table,
        ],
    )


def build_stat_tile(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        style={
            "backgroundColor": COLOR_SURFACE_ALT,
            "padding": "10px",
            "borderRadius": "8px",
            "minWidth": "200px",
            "color": COLOR_TEXT_PRIMARY,
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "color": COLOR_TEXT_MUTED}),
            html.Div(value, style={"fontSize": "18px", "fontWeight": "700", "marginTop": "4px"}),
            html.Div(
                subtitle,
                style={"fontSize": "11px", "color": COLOR_TEXT_SUBTLE, "marginTop": "2px"},
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
                        style={
                            "marginTop": "6px",
                            "backgroundColor": COLOR_ACCENT,
                            "color": "white",
                            "border": "none",
                            "padding": "6px 10px",
                            "borderRadius": "6px",
                            "fontSize": "11px",
                        },
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
                color="#4caf50",
                size=170,
            ),
            daq.Gauge(
                min=0,
                max=100,
                value=summary.get("pct_30d", 0.0),
                showCurrentValue=True,
                label="Within limits (last 30d)",
                color="#81c784",
                size=170,
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


# ----------------- dash app -----------------

app = Dash(__name__)
app.title = "ARC Reporting"
server = app.server

app.layout = html.Div(
    style={
        "backgroundColor": COLOR_BG,
        "color": COLOR_TEXT_PRIMARY,
        "fontFamily": "Segoe UI, sans-serif",
        "padding": "16px",
        "minHeight": "100vh",
    },
    children=[
        # Header
        html.Div(
            style={
                "backgroundColor": COLOR_SURFACE,
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
                            "ARC Reporting",
                            style={
                                "fontSize": "18px",
                                "fontWeight": "700",
                            },
                        ),
                        html.Div(
                            "Live view of PLC tags and hourly aggregates",
                            style={
                                "fontSize": "11px",
                                "color": COLOR_TEXT_MUTED,
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
                                "fontSize": "11px",
                                "color": COLOR_TEXT_SUBTLE,
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
        dcc.Download(id="export-minute-download"),
        dcc.Download(id="export-hourly-download"),
        dcc.Download(id="export-rolling-download"),
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

        # Tabs: Overview (gauges) & Thresholds (editor)
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            colors={
                "border": COLOR_BORDER,
                "primary": COLOR_ACCENT,
                "background": COLOR_SURFACE_ALT,
            },
            style={
                "borderRadius": "8px",
                "overflow": "hidden",
            },
            children=[
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    style={"backgroundColor": COLOR_SURFACE_ALT, "color": COLOR_TEXT_MUTED},
                    selected_style={
                        "backgroundColor": COLOR_SURFACE,
                        "color": COLOR_TEXT_PRIMARY,
                        "fontWeight": "600",
                    },
                    children=[
                        html.Div(
                            id="tag-cards-container",
                            style={
                                "padding": "12px",
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "16px",
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Thresholds",
                    value="thresholds",
                    style={"backgroundColor": COLOR_SURFACE_ALT, "color": COLOR_TEXT_MUTED},
                    selected_style={
                        "backgroundColor": COLOR_SURFACE,
                        "color": COLOR_TEXT_PRIMARY,
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
                                    style={"fontSize": "11px", "color": COLOR_TEXT_MUTED},
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
                                        "backgroundColor": COLOR_ACCENT,
                                        "color": "white",
                                        "border": "none",
                                        "padding": "6px 10px",
                                        "borderRadius": "6px",
                                        "fontSize": "11px",
                                    },
                                ),
                                html.Div(
                                    id="threshold-save-status",
                                    style={"fontSize": "11px", "color": COLOR_TEXT_SUBTLE},
                                ),
                                html.Div(
                                    f"File: {THRESHOLDS_JSON}",
                                    style={
                                        "marginTop": "8px",
                                        "fontSize": "9px",
                                        "color": COLOR_TEXT_SUBTLE,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Exports",
                    value="exports",
                    style={"backgroundColor": COLOR_SURFACE_ALT, "color": COLOR_TEXT_MUTED},
                    selected_style={
                        "backgroundColor": COLOR_SURFACE,
                        "color": COLOR_TEXT_PRIMARY,
                        "fontWeight": "600",
                    },
                    children=[
                        html.Div(
                            style={
                                "padding": "12px 16px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "12px",
                                "maxWidth": "760px",
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
                                                    "color": "#b0bec5",
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
                                                    "color": "#b0bec5",
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
                                                    "color": "#b0bec5",
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
        thresholds = load_thresholds()
        config_version = compute_config_version_text()
        alias_map = build_alias_lookup(
            raw_df,
            hourly_df,
            rolling_df,
            load_alias_map_from_settings(),
        )

        _, _, last_ts = extract_raw_stats(raw_df)
        last_full_hour_stats = extract_last_full_hour(hourly_df)
        rolling_12hr_stats = extract_latest_rolling_12hr(rolling_df)

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
        tags = [t for t in tags if not looks_numeric_tag(str(t))]

        cems_map = {
            "o2": "CEMS O2",
            "nox": "CEMS NOX",
            "co": "CEMS CO",
        }

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
        for metric, label in cems_map.items():
            tag = find_cems_tag(metric, tags, thresholds, alias_map)
            use_avg_value = metric == "o2"
            units_label = "%" if use_avg_value else "lb/hr"
            cards.append(
                build_cems_card(
                    label=label,
                    tag=tag,
                    current_hour_lb_hr=current_hour_lb_hr,
                    current_hour_avg=current_hour_avg,
                    rolling_12hr_stats=rolling_12hr_stats,
                    thresholds=thresholds,
                    alias_map=alias_map,
                    units_label=units_label,
                    use_avg_value=use_avg_value,
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
