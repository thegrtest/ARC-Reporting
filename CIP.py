#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIP Tag Poller & Dashboard (PySide6, dark themed, production-ready)

Features
--------
- Polls a list of PLC tags at a configurable interval (default 10s) using pylogix.
- Logs all raw samples to logs/raw_data_YYYY-MM-DD.csv:
    timestamp, date, time, tag, alias, value, status
- Aggregates data into hourly averages (1–2pm, 2–3pm, …) and writes to
  logs/hourly_averages.csv with UPSERT semantics:
  for each (hour_start, tag) there is at most one row (latest wins).
- Dark, professional GUI showing:
    Alias | Tag | Current Value | Last Hour Avg | Current Hour Avg (so far) | Status
- Button to compute current hour averages (display only).
- Button to recompute/rebuild hourly_averages.csv from all daily raw_data_*.csv.
- Settings (machine, IP, tags+aliases, interval, chunk size, log dir, machine-state tag,
  polling mode, heartbeat tag/mode) are remembered in settings.json.

Tag + Alias Input
-----------------
In the “Tags to Poll” box, each line may be either:
    Program:MainRoutine.MyTag1
    Program:MainRoutine.Flow_PV | Flow
The part after '|' (if present) is used as the alias. Aliases are optional.

Reliability / Always-on
-----------------------
- PollThread uses a reconnect loop: if PLC connect fails, it logs the error and
  retries after a delay, indefinitely, until you press Stop.
- Read errors are caught and logged, not fatal.
- If a multi-read hits "too many parameters", the thread automatically falls
  back to single-tag reads for that batch.
- Optional machine-state gating:
    - Reads a DINT machine state tag (e.g. RotaryKiln_OperationState).
    - If "Pause when machine down" is selected, the thread:
        * Always reads the state tag every cycle.
        * Only polls other tags when state == 3 (Processing).
        * Automatically resumes full polling when state returns to 3.
- Optional heartbeat writer:
    - Writes a toggling BOOL/DINT heartbeat every cycle to a configured tag.
    - If polling stops or connection is lost, heartbeat stops changing.
    - PLC should implement a watchdog on this tag to safely stop the machine
      if heartbeat has not changed within a configured timeout.
- No modal message boxes from the background thread, to avoid freezes.
"""

import sys
import os
import csv
import json
import time
import hashlib
import getpass
import gzip
import shutil
import traceback
from collections import deque
from datetime import datetime, timedelta, date
from typing import Dict, Tuple, List, Optional

from PySide6.QtCore import Qt, QThread, Signal, QMutex, QMutexLocker, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QPalette,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QPushButton, QTextEdit, QTableWidget,
    QTableWidgetItem, QMessageBox, QGroupBox, QFormLayout, QSpinBox,
    QFileDialog, QHeaderView, QPlainTextEdit, QStatusBar, QFrame,
    QSpacerItem, QSizePolicy, QComboBox, QCheckBox, QScrollArea, QTabWidget,
    QProgressBar, QSplashScreen, QDoubleSpinBox
)

try:
    from pylogix import PLC  # type: ignore
    PYLOGIX_AVAILABLE = True
except Exception:  # ImportError or others
    PYLOGIX_AVAILABLE = False


# Machine state enumeration for logging / readability
# 0=OFF, 1=Warm up, 2=Idle, 3=Processing, 4=Shutdown
MACHINE_STATE_MAP = {
    0: "OFF",
    1: "Warm up",
    2: "Idle",
    3: "Processing",
    4: "Shutdown",
}


# ------------------- helpers -------------------


def ensure_dir(path: str) -> None:
    d = os.path.abspath(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def ensure_csv(path: str, headers: List[str]) -> None:
    """Create CSV with headers if it does not exist."""
    ensure_dir(os.path.dirname(path) or ".")
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def _to_float(val: object) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_threshold_entry(raw_entry: object) -> Dict[str, object]:
    if not isinstance(raw_entry, dict):
        return {}

    entry: Dict[str, object] = {}
    for key in ["alias", "units"]:
        if isinstance(raw_entry.get(key), str) and raw_entry.get(key).strip():
            entry[key] = raw_entry.get(key).strip()

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


def chunked(seq: List[str], size: int):
    """Yield chunks from list (used to batch pylogix reads)."""
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


def hour_bucket(dt: datetime) -> datetime:
    """Return datetime truncated to hour (start of hour)."""
    return dt.replace(minute=0, second=0, microsecond=0)


def resource_path(name: str) -> str:
    """Return an absolute path for bundled assets (icons, etc.)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, name)


class GaugeDisplay(QWidget):
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(True)
        self.bar.setAlignment(Qt.AlignCenter)

        self.range_label = QLabel("")
        self.range_label.setStyleSheet("color: #90a4ae; font-size: 11px;")
        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: #b0bec5; font-size: 11px;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.bar)
        layout.addWidget(self.range_label)
        layout.addWidget(self.detail_label)

    def update_display(
        self,
        value: Optional[float],
        gauge_min: float,
        gauge_max: float,
        color: str,
        detail_text: str,
    ) -> None:
        if gauge_max <= gauge_min:
            gauge_max = gauge_min + 1.0

        try:
            disp_val = float(value) if value is not None else float("nan")
        except Exception:
            disp_val = float("nan")

        if disp_val != disp_val:  # NaN
            percent = 0
            formatted = "—"
        else:
            percent = int(1000 * (disp_val - gauge_min) / max(1e-6, gauge_max - gauge_min))
            percent = max(0, min(1000, percent))
            formatted = f"{disp_val:.2f}"

        self.bar.setValue(percent)
        self.bar.setFormat(formatted)
        self.bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid #2b323c;
                border-radius: 6px;
                background: #11151b;
                color: #e0e6ed;
                text-align: center;
                padding: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 6px;
            }}
            """
        )

        self.range_label.setText(f"Gauge range: {gauge_min:.2f} – {gauge_max:.2f}")
        self.detail_label.setText(detail_text)


class GaugeCard(QFrame):
    def __init__(self, tag: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tag = tag
        self.setObjectName("gaugeCard")
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(2)

        self.title_label = QLabel(tag)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        self.subtitle_label = QLabel("")
        self.subtitle_label.setStyleSheet("color: #90a4ae; font-size: 11px;")

        self.oper_label = QLabel("")
        self.oper_label.setStyleSheet("color: #b0bec5; font-size: 11px;")
        self.limit_label = QLabel("")
        self.limit_label.setStyleSheet("color: #b0bec5; font-size: 11px;")

        self.qa_label = QLabel("")
        self.qa_label.setStyleSheet("color: #ffb74d; font-size: 11px;")

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)
        header_layout.addWidget(self.oper_label)
        header_layout.addWidget(self.limit_label)
        header_layout.addWidget(self.qa_label)

        gauges_row = QHBoxLayout()
        gauges_row.setSpacing(12)

        self.current_display = GaugeDisplay("Current")
        self.last_display = GaugeDisplay("Last Full Hour")
        self.live_display = GaugeDisplay("Current Hour Avg")

        gauges_row.addWidget(self.current_display)
        gauges_row.addWidget(self.last_display)
        gauges_row.addWidget(self.live_display)

        layout.addLayout(header_layout)
        layout.addLayout(gauges_row)

    def update_metadata(
        self,
        display_name: str,
        tag: str,
        units: str,
        low_oper: Optional[float],
        high_oper: Optional[float],
        low_limit: Optional[float],
        high_limit: Optional[float],
    ) -> None:
        self.title_label.setText(display_name or tag)

        subtitle_bits = []
        if display_name and display_name != tag:
            subtitle_bits.append(f"Tag: {tag}")
        if units:
            subtitle_bits.append(f"Units: {units}")
        self.subtitle_label.setText(" • ".join(subtitle_bits))

        def format_bounds(label: str, low_val: Optional[float], high_val: Optional[float], fallback: str) -> str:
            bounds: List[str] = []
            if low_val is not None:
                bounds.append(str(low_val))
            if high_val is not None:
                bounds.append(str(high_val))
            return f"{label}: [{' – '.join(bounds)}]" if bounds else fallback

        self.oper_label.setText(
            format_bounds("Operational", low_oper, high_oper, "Operational: auto range")
        )
        self.limit_label.setText(
            format_bounds("Regulatory limit", low_limit, high_limit, "Regulatory limit: none set")
        )

    def update_values(
        self,
        current_val: Optional[float],
        last_avg: Optional[float],
        live_avg: Optional[float],
        gauge_min: float,
        gauge_max: float,
        current_color: str,
        last_color: str,
        live_color: str,
        current_detail: str,
        last_detail: str,
        live_detail: str,
    ) -> None:
        self.qa_label.setText(current_detail)
        self.current_display.update_display(current_val, gauge_min, gauge_max, current_color, current_detail)
        self.last_display.update_display(last_avg, gauge_min, gauge_max, last_color, last_detail)
        self.live_display.update_display(live_avg, gauge_min, gauge_max, live_color, live_detail)
# ------------------- background polling thread -------------------


class PollThread(QThread):
    """
    QThread that maintains a PLC connection and polls tags every interval.
    Emits data_ready(timestamp_iso, {tag: (value, status)}) on each cycle.

    This thread is designed to *never die* unless .stop() is called:
    - If connection fails, it logs an error and retries after reconnect_delay_sec.
    - If per-read errors occur, it logs and continues.
    - If a multi-read hits 'too many parameters', it falls back to
      single-tag reads automatically.
    - Optional machine-state gating: always read the configured state tag,
      and only poll main tags when state == 3 (Processing).
    - Optional heartbeat writer: writes a toggling BOOL/DINT every cycle
      to a configured heartbeat tag.
    """

    data_ready = Signal(str, dict)
    error = Signal(str)
    info = Signal(str)

    # Conservative upper bound for CIP multi-read requests.
    MAX_SAFE_CHUNK = 10

    def __init__(
        self,
        ip: str,
        tags: List[str],
        interval_sec: int = 10,
        chunk_size: int = 20,
        reconnect_delay_sec: int = 10,
        machine_state_tag: Optional[str] = None,
        pause_when_down: bool = False,
        heartbeat_tag: Optional[str] = None,
        heartbeat_enabled: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.ip = ip

        # main tags to poll (excluding machine-state logic)
        self.tags_main = [t for t in tags if t.strip()]

        self.interval_sec = max(1, int(interval_sec))

        # clamp chunk size to a safe upper bound
        user_chunk = max(1, int(chunk_size))
        if user_chunk > self.MAX_SAFE_CHUNK:
            self._clamped_chunk_msg = (
                f"Requested chunk_size={user_chunk}, clamped to "
                f"MAX_SAFE_CHUNK={self.MAX_SAFE_CHUNK} to avoid "
                f"'too many parameters' PLC errors."
            )
            self.chunk_size = self.MAX_SAFE_CHUNK
        else:
            self._clamped_chunk_msg = ""
            self.chunk_size = user_chunk

        # Machine state control
        self.machine_state_tag = (machine_state_tag or "").strip()
        self.pause_when_down = bool(pause_when_down)

        # Heartbeat control
        self.heartbeat_tag = (heartbeat_tag or "").strip()
        self.heartbeat_enabled = bool(heartbeat_enabled and self.heartbeat_tag)
        self._heartbeat_value = False  # toggled each cycle when enabled

        # How long to wait before reattempting a PLC connection after failure
        self.reconnect_delay_sec = max(1, int(reconnect_delay_sec))

        self._running = False
        self._mutex = QMutex()

        # For logging pause/resume transitions
        self._main_poll_paused = False

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    def _should_run(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    # ---------- internal helpers ----------

    def _multi_read_with_fallback(self, comm: "PLC", batch: List[str]) -> Dict[str, Tuple[object, str]]:
        """
        Try a multi-read (comm.Read(*batch)). If we hit a 'too many parameters'
        error from the PLC, automatically fall back to reading each tag
        individually.

        Returns:
            dict: {tag_name: (value, status)}
        """
        results: Dict[str, Tuple[object, str]] = {}

        # First attempt: multi-read
        try:
            res = comm.Read(*batch)
            if not isinstance(res, list):
                res = [res]
        except Exception as e:
            msg = str(e).lower()
            if "too many parameters" in msg or "too many attributes" in msg:
                self.info.emit(
                    "PLC reported 'too many parameters' on multi-read; "
                    "falling back to single-tag reads for this batch."
                )
            else:
                raise
        else:
            too_many_via_status = False
            for r in res:
                status_text = str(getattr(r, "Status", "")).lower()
                if "too many parameters" in status_text or "too many attributes" in status_text:
                    too_many_via_status = True
                    break

            if not too_many_via_status:
                for r in res:
                    tag = getattr(r, "TagName", "(unknown)")
                    val = getattr(r, "Value", None)
                    st = getattr(r, "Status", "(unknown)")
                    results[tag] = (val, st)
                return results

            self.info.emit(
                "PLC status indicated 'too many parameters' on multi-read; "
                "falling back to single-tag reads for this batch."
            )

        # Fallback path: single-tag reads
        for tag in batch:
            try:
                r = comm.Read(tag)
                if isinstance(r, list):
                    objs = r
                else:
                    objs = [r]
                for obj in objs:
                    t = getattr(obj, "TagName", tag)
                    val = getattr(obj, "Value", None)
                    st = getattr(obj, "Status", "(unknown)")
                    results[t] = (val, st)
            except Exception as e:
                self.error.emit(f"Single-tag read error for '{tag}': {e}")
                results[tag] = (None, f"Error: {e}")
        return results

    def _decode_machine_state(self, val) -> Optional[int]:
        """Try to coerce the PLC value to an int state code."""
        try:
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str) and val.strip() != "":
                return int(float(val))
        except Exception:
            return None
        return None

    def _write_heartbeat(self, comm: "PLC"):
        """
        Write a toggling heartbeat (0/1) to the configured heartbeat tag.
        Any errors are logged but do not break the polling loop.
        """
        if not self.heartbeat_enabled or not self.heartbeat_tag:
            return
        try:
            self._heartbeat_value = not self._heartbeat_value
            value_to_write = int(self._heartbeat_value)
            r = comm.Write(self.heartbeat_tag, value_to_write)
            status = getattr(r, "Status", "Unknown")
            if str(status).lower() != "success":
                self.error.emit(
                    f"Heartbeat write failed for '{self.heartbeat_tag}': {status}"
                )
        except Exception as e:
            self.error.emit(f"Heartbeat write error for '{self.heartbeat_tag}': {e}")

    # ---------- main thread loop ----------

    def run(self):
        if not PYLOGIX_AVAILABLE:
            self.error.emit("pylogix is not installed. Run: pip install pylogix")
            return

        if not self.tags_main and not self.machine_state_tag and not self.heartbeat_enabled:
            self.error.emit("No tags, machine state, or heartbeat configured.")
            return

        with QMutexLocker(self._mutex):
            self._running = True

        if getattr(self, "_clamped_chunk_msg", ""):
            self.info.emit(self._clamped_chunk_msg)

        # Always-on reconnect loop
        while self._should_run():
            try:
                with PLC() as comm:
                    comm.IPAddress = self.ip
                    self.info.emit(
                        f"Connected to PLC at {self.ip} "
                        f"({len(self.tags_main)} main tags, chunk_size={self.chunk_size}, "
                        f"machine_state_tag={self.machine_state_tag or 'None'}, "
                        f"pause_when_down={self.pause_when_down}, "
                        f"heartbeat_tag={self.heartbeat_tag or 'None'}, "
                        f"heartbeat_enabled={self.heartbeat_enabled})"
                    )

                    self._main_poll_paused = False

                    while self._should_run():
                        start_t = time.time()
                        all_results: Dict[str, Tuple[object, str]] = {}
                        current_state_val: Optional[int] = None

                        # 1) Always read machine state tag first (if configured)
                        if self.machine_state_tag:
                            try:
                                state_dict = self._multi_read_with_fallback(
                                    comm, [self.machine_state_tag]
                                )
                                all_results.update(state_dict)
                                if self.machine_state_tag in state_dict:
                                    v, st = state_dict[self.machine_state_tag]
                                    if str(st).lower() == "success":
                                        current_state_val = self._decode_machine_state(v)
                            except Exception as e:
                                self.error.emit(
                                    f"State tag read error '{self.machine_state_tag}': {e}"
                                )

                        # 2) Decide whether to pause main polling
                        pause_main = False
                        if (
                            self.pause_when_down
                            and current_state_val is not None
                        ):
                            if current_state_val != 3:  # 3 == Processing
                                pause_main = True

                        # 3) Log transitions pause<->resume
                        if pause_main != self._main_poll_paused:
                            self._main_poll_paused = pause_main
                            state_name = (
                                MACHINE_STATE_MAP.get(current_state_val, str(current_state_val))
                                if current_state_val is not None
                                else "Unknown"
                            )
                            if pause_main:
                                self.info.emit(
                                    f"Machine state={current_state_val} ({state_name}); "
                                    "pausing main tag polling (state-only + heartbeat mode)."
                                )
                            else:
                                self.info.emit(
                                    f"Machine state={current_state_val} ({state_name}); "
                                    "resuming full tag polling."
                                )

                        # 4) If not paused, read main tags
                        try:
                            if (not pause_main) and self.tags_main:
                                main_tags = [
                                    t for t in self.tags_main
                                    if t != self.machine_state_tag
                                ]
                                for batch in chunked(main_tags, self.chunk_size):
                                    if not batch:
                                        continue
                                    batch_results = self._multi_read_with_fallback(
                                        comm, batch
                                    )
                                    all_results.update(batch_results)
                        except Exception as e:
                            self.error.emit(f"Read error: {e}")
                            time.sleep(self.interval_sec)
                            continue

                        # 5) Heartbeat write
                        self._write_heartbeat(comm)

                        ts_iso = datetime.now().isoformat(timespec="seconds")
                        self.data_ready.emit(ts_iso, all_results)

                        elapsed = time.time() - start_t
                        sleep_for = max(0.0, self.interval_sec - elapsed)
                        end_time = time.time() + sleep_for
                        while time.time() < end_time:
                            if not self._should_run():
                                break
                            time.sleep(0.1)

            except Exception as e:
                self.error.emit(f"PLC connection error: {e}")
                end_time = time.time() + self.reconnect_delay_sec
                while time.time() < end_time:
                    if not self._should_run():
                        break
                    time.sleep(0.5)

        self.info.emit("Polling thread stopped.")


class WritebackThread(QThread):
    """
    Writes current-hour averages back to PLC tags at a configured interval.
    Uses a separate PLC connection from the polling thread.
    """

    info = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        ip: str,
        interval_sec: int,
        writeback_map: Dict[str, str],
        reconnect_delay_sec: int = 10,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.ip = ip
        self.interval_sec = max(1, int(interval_sec))
        self.writeback_map = dict(writeback_map)
        self.reconnect_delay_sec = max(1, int(reconnect_delay_sec))

        self._running = False
        self._mutex = QMutex()
        self._avg_mutex = QMutex()
        self._current_averages: Dict[str, float] = {}

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    def _should_run(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def update_averages(self, averages: Dict[str, float]) -> None:
        with QMutexLocker(self._avg_mutex):
            self._current_averages = dict(averages)

    def _snapshot_averages(self) -> Dict[str, float]:
        with QMutexLocker(self._avg_mutex):
            return dict(self._current_averages)

    def _write_values(self, comm: "PLC", averages: Dict[str, float]) -> None:
        for source_tag, target_tag in self.writeback_map.items():
            if source_tag not in averages:
                continue
            value = averages.get(source_tag)
            if value is None:
                continue
            try:
                r = comm.Write(target_tag, float(value))
                status = getattr(r, "Status", "Unknown")
                if str(status).lower() != "success":
                    self.error.emit(
                        f"Writeback failed for '{source_tag}' -> '{target_tag}': {status}"
                    )
            except Exception as e:
                self.error.emit(
                    f"Writeback error for '{source_tag}' -> '{target_tag}': {e}"
                )

    def run(self):
        if not PYLOGIX_AVAILABLE:
            self.error.emit("pylogix is not installed. Run: pip install pylogix")
            return

        if not self.writeback_map:
            self.error.emit("Writeback is enabled but no mappings are configured.")
            return

        with QMutexLocker(self._mutex):
            self._running = True

        while self._should_run():
            try:
                with PLC() as comm:
                    comm.IPAddress = self.ip
                    self.info.emit(
                        f"Connected to PLC at {self.ip} for hourly average writeback "
                        f"({len(self.writeback_map)} mapping(s), interval={self.interval_sec}s)."
                    )

                    while self._should_run():
                        start_t = time.time()
                        averages = self._snapshot_averages()
                        if averages:
                            self._write_values(comm, averages)

                        elapsed = time.time() - start_t
                        sleep_for = max(0.0, self.interval_sec - elapsed)
                        end_time = time.time() + sleep_for
                        while time.time() < end_time:
                            if not self._should_run():
                                break
                            time.sleep(0.1)
            except Exception as e:
                self.error.emit(f"Writeback PLC connection error: {e}")
                end_time = time.time() + self.reconnect_delay_sec
                while time.time() < end_time:
                    if not self._should_run():
                        break
                    time.sleep(0.5)

        self.info.emit("Writeback thread stopped.")


# ------------------- main window -------------------


class MainWindow(QMainWindow):
    SETTINGS_FILE = "settings.json"
    CONFIG_CHANGE_HEADERS = [
        "timestamp",
        "user",
        "field",
        "old_value",
        "new_value",
        "reason",
    ]
    EPA19_STD_O2_PCT = 20.9
    EPA19_MOLAR_VOLUME_SCF = 385.8
    EPA19_MOLECULAR_WEIGHTS = {
        "NOx": 46.0,  # as NO2
        "CO": 28.01,
        "O2": 32.0,
    }

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CIP Tag Poller & Dashboard")
        if QApplication.instance() and not QApplication.instance().windowIcon().isNull():
            self.setWindowIcon(QApplication.instance().windowIcon())
        self._apply_initial_size()

        # --- state ---
        self.poll_thread: Optional[PollThread] = None
        self.writeback_thread: Optional[WritebackThread] = None
        self.current_values: Dict[str, Tuple[object, str]] = {}
        self.last_hour_avg: Dict[str, float] = {}
        self.current_hour_preview: Dict[str, float] = {}
        self.current_hour_start: Optional[datetime] = None
        self.hour_accumulators: Dict[str, Dict[str, float]] = {}
        self.last_success_ts: Dict[str, datetime] = {}
        self.last_poll_success_ts_iso: Optional[str] = None
        self.last_poll_error_ts_iso: Optional[str] = None
        self.error_timestamps = deque()  # rolling window for last hour
        self._stale_tracker: Dict[str, Dict[str, object]] = {}
        self._log_size_warned = False
        self.lockdown_enabled = False
        self.gauge_cards: Dict[str, GaugeCard] = {}
        self._gauge_spacer: Optional[QSpacerItem] = None
        self.writeback_interval_sec = 60
        self.units_map: Dict[str, str] = {}

        # stable tag ordering for the dashboard
        self.tag_order: List[str] = []
        # tag -> alias mapping
        self.alias_map: Dict[str, str] = {}
        # last loaded settings snapshot for change detection
        self._last_settings_snapshot: Dict[str, object] = {}

        # data quality controls
        self.stale_intervals_threshold = 5
        self.gap_threshold_minutes = 5
        self.moving_tags: List[str] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}

        # default config
        self.machine_name = ""
        self.epa_enabled = False
        self.epa_flow_tag = ""
        self.epa_o2_tag = ""
        self.epa_o2_units = "percent"
        self.epa_ref_o2_pct = 3.0
        self.epa_nox_tag = ""
        self.epa_co_tag = ""
        self.log_dir = os.path.join("logs")
        self.raw_csv_path = ""   # will be set based on current date
        self.current_log_date: date = datetime.now().date()
        self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")
        self.env_events_path = os.path.join(self.log_dir, "env_events.csv")
        self.system_health_path = os.path.join(self.log_dir, "system_health.json")
        self.log_dir_warn_gb = 5.0
        self.disk_warn_gb = 2.0
        self.disk_crit_gb = 1.0
        self.compress_raw_after_days = 7

        # UI
        self._build_ui()
        self._apply_dark_theme()
        self._load_settings()  # populates fields and paths
        self._update_config_version_label()

        # init daily raw path
        self.current_log_date = datetime.now().date()
        self.raw_csv_path = self._ensure_raw_csv_for_date(self.current_log_date)

        # ensure hourly CSV
        ensure_csv(
            self.hourly_csv_path,
            ["hour_start", "hour_end", "tag", "avg_value", "avg_lb_hr", "sample_count"],
        )

    def _apply_initial_size(self) -> None:
        """Scale the window to fit the available screen while remaining sizable."""
        screen = QApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            target_width = int(available.width() * 0.9)
            target_height = int(available.height() * 0.9)
            self.resize(target_width, target_height)

            frame_geo = self.frameGeometry()
            frame_geo.moveCenter(available.center())
            self.move(frame_geo.topLeft())
        else:
            self.resize(1300, 750)

    # -------- tag + alias parsing --------

    @staticmethod
    def parse_tags_and_aliases(text: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Parse tags text area into:
            tags:      [tag1, tag2, ...]
            alias_map: {tag: alias}
        Syntax per line:
            TAG
            TAG | Alias
        """
        tags: List[str] = []
        alias_map: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                tag_part, alias_part = line.split("|", 1)
                tag = tag_part.strip()
                alias = alias_part.strip()
            else:
                tag = line
                alias = ""
            if not tag:
                continue
            tags.append(tag)
            if alias:
                alias_map[tag] = alias
        return tags, alias_map

    @staticmethod
    def parse_writeback_mappings(text: str) -> Dict[str, str]:
        """
        Parse writeback mappings into {alias: plc_tag}.
        Accepts lines like:
            Alias | PLC_Tag
            Alias -> PLC_Tag
        """
        mappings: Dict[str, str] = {}
        for line in text.splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if "|" in raw:
                alias_part, tag_part = raw.split("|", 1)
            elif "->" in raw:
                alias_part, tag_part = raw.split("->", 1)
            else:
                continue
            alias = alias_part.strip()
            tag = tag_part.strip()
            if not alias or not tag:
                continue
            mappings[alias] = tag
        return mappings

    def _resolve_writeback_targets(self, mappings: Dict[str, str]) -> Dict[str, str]:
        """Resolve alias mappings into {source_tag: plc_tag} using current alias map."""
        if not mappings:
            return {}
        alias_to_tag = {alias: tag for tag, alias in self.alias_map.items() if alias}
        resolved: Dict[str, str] = {}
        for alias, target_tag in mappings.items():
            source_tag = alias_to_tag.get(alias)
            if not source_tag:
                if alias in self.tag_order or alias in self.alias_map:
                    source_tag = alias
            if not source_tag:
                self.log_message(
                    f"Writeback mapping skipped: alias '{alias}' not found in configured tags."
                )
                continue
            resolved[source_tag] = target_tag
        return resolved

    # -------- raw CSV path helpers (daily rotation) --------

    def _raw_path_for_date(self, d: date) -> str:
        """Return raw_data CSV path for a given date."""
        fname = f"raw_data_{d.isoformat()}.csv"
        return os.path.join(self.log_dir, fname)

    def _ensure_raw_csv_for_date(self, d: date) -> str:
        """Ensure daily raw CSV exists for given date; return its path."""
        path = self._raw_path_for_date(d)
        ensure_csv(
            path,
            ["timestamp", "date", "time", "tag", "alias", "value", "status", "qa_flag"],
        )
        return path

    def _ensure_env_events_csv(self) -> None:
        ensure_csv(
            self.env_events_path,
            ["timestamp", "event_type", "tag", "duration_sec"],
        )

    def _load_thresholds_from_log_dir(self) -> Dict[str, Dict[str, float]]:
        path = os.path.join(self.log_dir, "thresholds.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: _normalize_threshold_entry(v) for k, v in data.items()}
        except Exception:
            return {}
        return {}

    # -------- system health & storage --------

    def _compute_disk_free_gb(self) -> float:
        try:
            usage = shutil.disk_usage(self.log_dir)
            return round(usage.free / (1024 ** 3), 2)
        except Exception:
            return 0.0

    def _compute_log_dir_size_gb(self) -> float:
        total_bytes = 0
        for root, _, files in os.walk(self.log_dir):
            for name in files:
                try:
                    total_bytes += os.path.getsize(os.path.join(root, name))
                except Exception:
                    continue
        return round(total_bytes / (1024 ** 3), 2)

    def _compress_old_raw_csvs(self) -> None:
        if self.compress_raw_after_days <= 0:
            return
        cutoff = datetime.now().date() - timedelta(days=self.compress_raw_after_days)
        try:
            for name in os.listdir(self.log_dir):
                if not name.startswith("raw_data_"):
                    continue
                if name.endswith(".csv.gz"):
                    continue
                if not name.endswith(".csv"):
                    continue
                try:
                    date_part = name[len("raw_data_") : -len(".csv")]
                    file_date = datetime.fromisoformat(date_part).date()
                except Exception:
                    continue
                if file_date >= cutoff:
                    continue
                src_path = os.path.join(self.log_dir, name)
                dst_path = src_path + ".gz"
                try:
                    with open(src_path, "rb") as src, gzip.open(dst_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.remove(src_path)
                    self.log_message(f"Compressed old raw CSV: {name} -> {os.path.basename(dst_path)}")
                except Exception as e:
                    self.log_message(f"Failed to compress {name}: {e}")
        except Exception as e:
            self.log_message(f"Error compressing old raw CSVs: {e}")

    def _determine_health_status(
        self,
        now: datetime,
        disk_free_gb: float,
        log_size_gb: float,
        error_count: int,
    ) -> Tuple[str, str]:
        status = "System Healthy"
        reasons: List[str] = []

        if self.last_poll_success_ts_iso:
            try:
                last_success_dt = datetime.fromisoformat(self.last_poll_success_ts_iso)
                age = now - last_success_dt
                if age > timedelta(minutes=5):
                    status = "Critical"
                    reasons.append("Polling has been stale for over 5 minutes")
                elif age > timedelta(minutes=2):
                    status = "Degraded"
                    reasons.append("Polling delayed beyond 2 minutes")
            except Exception:
                reasons.append("Unable to parse last poll timestamp")
        else:
            status = "Degraded"
            reasons.append("No successful poll recorded yet")

        if disk_free_gb <= self.disk_crit_gb:
            status = "Critical"
            reasons.append(f"Disk free critically low ({disk_free_gb} GB)")
        elif disk_free_gb <= self.disk_warn_gb and status != "Critical":
            status = "Degraded"
            reasons.append(f"Disk free low ({disk_free_gb} GB)")

        if log_size_gb >= self.log_dir_warn_gb and status != "Critical":
            status = "Degraded"
            reasons.append(
                f"Log folder size high ({log_size_gb} GB >= {self.log_dir_warn_gb} GB)"
            )

        if error_count >= 10:
            status = "Critical"
            reasons.append(f"{error_count} errors in last hour")
        elif error_count >= 3 and status != "Critical":
            status = "Degraded"
            reasons.append(f"{error_count} errors in last hour")

        reason_text = "; ".join(reasons) if reasons else "OK"
        return status, reason_text

    def _update_system_health(self, success_ts: Optional[datetime] = None, error: bool = False):
        now = success_ts or datetime.now()
        if success_ts:
            self.last_poll_success_ts_iso = success_ts.isoformat(timespec="seconds")
        if error:
            self.last_poll_error_ts_iso = now.isoformat(timespec="seconds")
            self.error_timestamps.append(now)

        cutoff = now - timedelta(hours=1)
        while self.error_timestamps and self.error_timestamps[0] < cutoff:
            self.error_timestamps.popleft()
        error_count = len(self.error_timestamps)

        disk_free_gb = self._compute_disk_free_gb()
        log_size_gb = self._compute_log_dir_size_gb()

        if log_size_gb >= self.log_dir_warn_gb and not self._log_size_warned:
            self._log_size_warned = True
            self.log_message(
                f"Warning: log directory size is {log_size_gb} GB (threshold {self.log_dir_warn_gb} GB)."
            )

        self._compress_old_raw_csvs()

        status, reason = self._determine_health_status(now, disk_free_gb, log_size_gb, error_count)

        health = {
            "last_poll_success_ts": self.last_poll_success_ts_iso,
            "last_poll_error_ts": self.last_poll_error_ts_iso,
            "error_count_last_hour": error_count,
            "disk_free_GB": disk_free_gb,
            "log_dir_size_GB": log_size_gb,
            "status": status,
            "status_reason": reason,
        }

        try:
            ensure_dir(self.log_dir)
            with open(self.system_health_path, "w", encoding="utf-8") as f:
                json.dump(health, f, indent=2)
        except Exception as e:
            self.log_message(f"Failed to write system health file: {e}")

    # ---------------- UI construction ----------------

    def _build_ui(self):
        central = QWidget()

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        # ===== Header bar =====
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(20)

        self.title_label = QLabel("CIP Tag Poller & Dashboard")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        self.machine_label = QLabel("Machine: —")
        machine_font = QFont()
        machine_font.setPointSize(11)
        self.machine_label.setFont(machine_font)

        self.config_version_label = QLabel("Config: —")
        config_font = QFont()
        config_font.setPointSize(10)
        self.config_version_label.setFont(config_font)

        header_left = QVBoxLayout()
        header_left.addWidget(self.title_label)
        header_left.addWidget(self.machine_label)
        header_left.addWidget(self.config_version_label)

        header_right = QVBoxLayout()
        header_right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.connection_status_label = QLabel("Status: Idle")
        self.connection_status_label.setObjectName("connectionStatusLabel")
        self.last_update_label = QLabel("Last update: —")
        self.last_update_label.setObjectName("lastUpdateLabel")

        header_right.addWidget(self.connection_status_label, alignment=Qt.AlignRight)
        header_right.addWidget(self.last_update_label, alignment=Qt.AlignRight)

        header_layout.addLayout(header_left, stretch=1)
        header_layout.addLayout(header_right, stretch=0)

        root_layout.addWidget(header_frame)

        # ===== Main content area =====
        main_content = QHBoxLayout()
        main_content.setSpacing(10)

        # ---- Left column: config + controls + log ----
        left_frame = QFrame()
        left_frame.setObjectName("cardFrame")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # Configuration group
        cfg_group = QGroupBox("Configuration")
        cfg_group.setObjectName("cardGroupBox")
        cfg_layout = QFormLayout(cfg_group)
        cfg_layout.setLabelAlignment(Qt.AlignLeft)
        cfg_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        cfg_layout.setHorizontalSpacing(10)
        cfg_layout.setVerticalSpacing(6)
        cfg_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Run-lock option
        self.lockdown_checkbox = QCheckBox(
            "Run — settings locked, auto-start on launch"
        )
        self.lockdown_checkbox.setToolTip(
            "When enabled, the app immediately starts polling with the saved settings, "
            "disables configuration changes, and cannot be stopped or closed."
        )
        self.lockdown_checkbox.setStyleSheet(
            "QCheckBox { font-weight: 700; color: #ffb74d; border: 1px solid #ffb74d; "
            "padding: 6px; border-radius: 6px; }"
        )
        cfg_layout.addRow(self.lockdown_checkbox)

        # Machine name
        self.machine_name_edit = QLineEdit()
        self.machine_name_edit.setPlaceholderText("e.g. CF Duplex 15")
        cfg_layout.addRow(QLabel("Machine Name:"), self.machine_name_edit)

        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("192.168.1.20")
        cfg_layout.addRow(QLabel("PLC IP Address:"), self.ip_edit)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 3600)
        self.interval_spin.setValue(10)
        cfg_layout.addRow(QLabel("Poll Interval (seconds):"), self.interval_spin)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 200)
        self.chunk_spin.setValue(20)
        cfg_layout.addRow(QLabel("Chunk Size (tags per read):"), self.chunk_spin)

        self.stale_spin = QSpinBox()
        self.stale_spin.setRange(2, 1000)
        self.stale_spin.setValue(self.stale_intervals_threshold)
        cfg_layout.addRow(QLabel("Stale after N identical reads:"), self.stale_spin)

        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(1, 1440)
        self.gap_spin.setValue(self.gap_threshold_minutes)
        cfg_layout.addRow(QLabel("Data gap threshold (minutes):"), self.gap_spin)

        self.moving_tags_edit = QLineEdit()
        self.moving_tags_edit.setPlaceholderText("Comma-separated tag names for stale detection")
        cfg_layout.addRow(QLabel("Tags expected to move:"), self.moving_tags_edit)

        epa_label = QLabel("EPA Method 19 (PPMV → lb/hr)")
        epa_label.setStyleSheet("font-weight: 600; color: #b0bec5; padding-top: 6px;")
        cfg_layout.addRow(epa_label)

        self.epa_enabled_checkbox = QCheckBox("Enable EPA Method 19 conversions")
        self.epa_enabled_checkbox.setToolTip(
            "Calculates lb/hr from PPMV using EPA Method 19 with a dry standard flow rate."
        )
        cfg_layout.addRow(self.epa_enabled_checkbox)

        self.epa_flow_tag_edit = QLineEdit()
        self.epa_flow_tag_edit.setPlaceholderText("e.g. Program:MainRoutine.StackFlow_DSCFM")
        cfg_layout.addRow(QLabel("Air Flow Tag (dscfm):"), self.epa_flow_tag_edit)

        self.epa_o2_tag_edit = QLineEdit()
        self.epa_o2_tag_edit.setPlaceholderText("e.g. Program:MainRoutine.O2_PV")
        cfg_layout.addRow(QLabel("O2 Tag (for correction):"), self.epa_o2_tag_edit)

        self.epa_o2_units_combo = QComboBox()
        self.epa_o2_units_combo.addItem("% O2", "percent")
        self.epa_o2_units_combo.addItem("PPMV", "ppmv")
        cfg_layout.addRow(QLabel("O2 Units:"), self.epa_o2_units_combo)

        self.epa_ref_o2_spin = QDoubleSpinBox()
        self.epa_ref_o2_spin.setRange(0.0, 20.9)
        self.epa_ref_o2_spin.setSingleStep(0.1)
        self.epa_ref_o2_spin.setDecimals(2)
        self.epa_ref_o2_spin.setValue(self.epa_ref_o2_pct)
        cfg_layout.addRow(QLabel("Reference O2 (%):"), self.epa_ref_o2_spin)

        self.epa_nox_tag_edit = QLineEdit()
        self.epa_nox_tag_edit.setPlaceholderText("e.g. Program:MainRoutine.NOx_PPMV")
        cfg_layout.addRow(QLabel("NOx Tag (PPMV):"), self.epa_nox_tag_edit)

        self.epa_co_tag_edit = QLineEdit()
        self.epa_co_tag_edit.setPlaceholderText("e.g. Program:MainRoutine.CO_PPMV")
        cfg_layout.addRow(QLabel("CO Tag (PPMV):"), self.epa_co_tag_edit)

        # Machine state tag + behavior
        self.machine_state_tag_edit = QLineEdit()
        self.machine_state_tag_edit.setPlaceholderText(
            "e.g. Program:MainRoutine.RotaryKiln_OperationState"
        )
        cfg_layout.addRow(QLabel("Machine State Tag (DINT):"), self.machine_state_tag_edit)

        self.poll_mode_combo = QComboBox()
        self.poll_mode_combo.addItems([
            "Poll always",
            "Pause when machine down",
        ])
        cfg_layout.addRow(QLabel("Polling Mode:"), self.poll_mode_combo)

        # Heartbeat config
        self.heartbeat_tag_edit = QLineEdit()
        self.heartbeat_tag_edit.setPlaceholderText(
            "e.g. Program:MainRoutine.RotaryKiln_Heartbeat"
        )
        cfg_layout.addRow(QLabel("Heartbeat Tag (BOOL/DINT):"), self.heartbeat_tag_edit)

        self.heartbeat_mode_combo = QComboBox()
        self.heartbeat_mode_combo.addItems([
            "Disabled",
            "Enabled",
        ])
        cfg_layout.addRow(QLabel("Heartbeat Mode:"), self.heartbeat_mode_combo)

        # Writeback config
        self.writeback_enabled_checkbox = QCheckBox(
            "Enable writeback of current hour averages"
        )
        self.writeback_enabled_checkbox.setToolTip(
            "When enabled, the app writes the current-hour average for each "
            "mapped alias back to its PLC tag at the configured interval."
        )
        cfg_layout.addRow(self.writeback_enabled_checkbox)

        self.writeback_interval_spin = QSpinBox()
        self.writeback_interval_spin.setRange(1, 3600)
        self.writeback_interval_spin.setValue(self.writeback_interval_sec)
        cfg_layout.addRow(QLabel("Writeback Interval (seconds):"), self.writeback_interval_spin)

        writeback_label = QLabel("Writeback Mappings (Alias | PLC Tag):")
        self.writeback_mappings_edit = QTextEdit()
        self.writeback_mappings_edit.setMinimumHeight(110)
        self.writeback_mappings_edit.setPlaceholderText(
            "Examples:\n"
            "Temperature | Program:MainRoutine.Temp_Avg_PLC\n"
            "Pressure | Program:MainRoutine.Pressure_Avg_PLC\n"
            "\n"
            "Each line:  Alias  |  PLC Tag to receive current-hour average"
        )
        cfg_layout.addRow(writeback_label, self.writeback_mappings_edit)

        tags_label = QLabel("Tags to Poll (one per line):")
        self.tags_edit = QTextEdit()
        self.tags_edit.setMinimumHeight(160)
        self.tags_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tags_edit.setPlaceholderText(
            "Examples:\n"
            "Program:MainRoutine.Temp_PV | Temperature\n"
            "Program:MainRoutine.Pressure_PV | Pressure\n"
            "Program:MainRoutine.Flow_PV\n"
            "\n"
            "Each line:  TAG  or  TAG | Alias"
        )
        cfg_layout.addRow(tags_label, self.tags_edit)

        # Log directory selector
        dir_layout = QHBoxLayout()
        self.log_dir_edit = QLineEdit(self.log_dir)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setObjectName("secondaryButton")
        dir_layout.addWidget(self.log_dir_edit)
        dir_layout.addWidget(self.browse_btn)
        cfg_layout.addRow(QLabel("Log Directory:"), dir_layout)

        left_layout.addWidget(cfg_group)

        # Controls
        controls_frame = QFrame()
        controls_frame.setObjectName("subCardFrame")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(8)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primaryButton")

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setEnabled(False)

        self.calc_btn = QPushButton("Current Hour Avg")
        self.calc_btn.setObjectName("secondaryButton")

        self.rebuild_btn = QPushButton("Rebuild Hourly CSV")
        self.rebuild_btn.setObjectName("secondaryButton")

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.calc_btn)
        controls_layout.addWidget(self.rebuild_btn)

        left_layout.addWidget(controls_frame)

        # Event log
        log_group = QGroupBox("Event Log")
        log_group.setObjectName("cardGroupBox")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        left_layout.addWidget(log_group, stretch=1)

        # Bottom spacer
        left_layout.addSpacerItem(
            QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        main_content.addWidget(left_frame, stretch=4)

        # ---- Right column: dashboard ----
        right_frame = QFrame()
        right_frame.setObjectName("cardFrame")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        self.dashboard_tabs = QTabWidget()
        self.dashboard_tabs.setObjectName("dashboardTabs")

        # ----- Table tab -----
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(12)

        dash_header_layout = QHBoxLayout()
        dash_title = QLabel("Tag Dashboard")
        dash_title_font = QFont()
        dash_title_font.setPointSize(13)
        dash_title_font.setBold(True)
        dash_title.setFont(dash_title_font)

        self.rows_summary_label = QLabel("Rows: 0")
        self.rows_summary_label.setObjectName("rowsSummaryLabel")

        dash_header_layout.addWidget(dash_title, alignment=Qt.AlignLeft)
        dash_header_layout.addStretch(1)
        dash_header_layout.addWidget(self.rows_summary_label, alignment=Qt.AlignRight)

        table_layout.addLayout(dash_header_layout)

        self.table = QTableWidget()
        # Alias + Tag + 5 metrics + QA flag
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            [
                "Alias",
                "Tag",
                "Current Value",
                "Last Hour Avg",
                "Current Hour Avg (so far)",
                "Last Hour Avg (lb/hr)",
                "Current Hour Avg (lb/hr)",
                "QA Flag",
            ]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Alias
        header.setSectionResizeMode(1, QHeaderView.Stretch)           # Tag
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)

        table_layout.addWidget(self.table, stretch=1)

        # ----- Gauges tab -----
        gauges_tab = QWidget()
        gauges_layout = QVBoxLayout(gauges_tab)
        gauges_layout.setContentsMargins(0, 0, 0, 0)
        gauges_layout.setSpacing(10)

        gauges_header = QLabel("Gauges (Current, Last Hour, Current Hour Avg)")
        gauges_header_font = QFont()
        gauges_header_font.setPointSize(13)
        gauges_header_font.setBold(True)
        gauges_header.setFont(gauges_header_font)

        gauges_layout.addWidget(gauges_header)

        self.gauge_scroll = QScrollArea()
        self.gauge_scroll.setWidgetResizable(True)
        self.gauge_container = QWidget()
        self.gauge_list_layout = QVBoxLayout(self.gauge_container)
        self.gauge_list_layout.setSpacing(12)
        self.gauge_list_layout.setContentsMargins(4, 4, 4, 4)
        self._gauge_spacer = QSpacerItem(
            20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding
        )
        self.gauge_list_layout.addItem(self._gauge_spacer)
        self.gauge_scroll.setWidget(self.gauge_container)
        gauges_layout.addWidget(self.gauge_scroll)

        self.dashboard_tabs.addTab(table_tab, "Table")
        self.dashboard_tabs.addTab(gauges_tab, "Gauges")

        right_layout.addWidget(self.dashboard_tabs, stretch=1)

        main_content.addWidget(right_frame, stretch=7)

        root_layout.addLayout(main_content)

        # Status bar
        status = QStatusBar()
        self.setStatusBar(status)

        # Connections
        self.start_btn.clicked.connect(self.start_polling)
        self.stop_btn.clicked.connect(self.stop_polling)
        self.calc_btn.clicked.connect(self.compute_current_hour_preview)
        self.rebuild_btn.clicked.connect(self.rebuild_hourly_from_raw)
        self.browse_btn.clicked.connect(self.choose_log_dir)
        self.lockdown_checkbox.stateChanged.connect(self._apply_lockdown_state)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        self.menuBar().addAction(exit_action)

    def _apply_dark_theme(self):
        app = QApplication.instance()
        if app is None:
            return

        app.setStyle("Fusion")
        palette = QPalette()

        palette.setColor(QPalette.Window, QColor(22, 25, 30))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(18, 20, 24))
        palette.setColor(QPalette.AlternateBase, QColor(30, 34, 40))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(32, 36, 42))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
        palette.setColor(QPalette.HighlightedText, Qt.white)

        app.setPalette(palette)

        app.setStyleSheet(
            """
            QMainWindow {
                background-color: #16191f;
            }
            #headerFrame {
                background-color: #20242b;
                border-radius: 8px;
            }
            #cardFrame {
                background-color: #1c2026;
                border-radius: 10px;
                border: 1px solid #2b323c;
            }
            #subCardFrame {
                background-color: #222730;
                border-radius: 8px;
                border: 1px solid #2b323c;
            }
            QGroupBox#cardGroupBox {
                border: 1px solid #2b323c;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                font-weight: 600;
            }
            QGroupBox#cardGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 4px;
                color: #b0bec5;
            }
            QLabel {
                color: #e0e6ed;
            }
            QLabel#connectionStatusLabel {
                font-weight: 600;
                color: #a5d6a7;
            }
            QLabel#lastUpdateLabel {
                color: #9e9e9e;
                font-size: 11px;
            }
            QLabel#rowsSummaryLabel {
                color: #b0bec5;
                font-size: 11px;
            }
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {
                background-color: #181c22;
                border: 1px solid #2b323c;
                border-radius: 4px;
                padding: 4px 6px;
                color: #e0e6ed;
                selection-background-color: #4caf50;
            }
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
            QSpinBox:focus, QComboBox:focus {
                border: 1px solid #4caf50;
            }
            QTableWidget {
                gridline-color: #2b323c;
                background-color: #181c22;
                alternate-background-color: #1f252e;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #252b35;
                color: #e0e6ed;
                padding: 6px 4px;
                border: 1px solid #2b323c;
                font-weight: 600;
            }
            QTableWidget::item {
                padding: 4px;
            }

            QPushButton {
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 500;
                color: #e0e6ed;
                background-color: #2c3139;
            }
            QPushButton:hover {
                background-color: #353b46;
            }
            QPushButton:pressed {
                background-color: #282d35;
            }
            QTabWidget::pane {
                border: 1px solid #2b323c;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #1c2026;
                color: #e0e6ed;
                padding: 8px 14px;
                border: 1px solid #2b323c;
                border-bottom-color: #2b323c;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #252b35;
            }
            QTabBar::tab:hover {
                background: #2d3441;
            }
            #gaugeCard {
                background-color: #1c2026;
                border-radius: 10px;
                border: 1px solid #2b323c;
            }

            QPushButton#primaryButton {
                background-color: #43a047;
            }
            QPushButton#primaryButton:hover {
                background-color: #4caf50;
            }
            QPushButton#primaryButton:pressed {
                background-color: #388e3c;
            }

            QPushButton#dangerButton {
                background-color: #e53935;
            }
            QPushButton#dangerButton:hover {
                background-color: #ef5350;
            }
            QPushButton#dangerButton:pressed {
                background-color: #c62828;
            }

            QPushButton#secondaryButton {
                background-color: #2f3a45;
            }
            QPushButton#secondaryButton:hover {
                background-color: #3a4653;
            }

            QScrollBar:vertical {
                background: #16191f;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3a414d;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4a5464;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    # ---------------- gauge helpers ----------------

    @staticmethod
    def _status_color(status: str) -> str:
        if status == "good":
            return "#4caf50"
        if status == "warning":
            return "#fdd835"
        if status == "bad":
            return "#ef5350"
        return "#78909c"

    @staticmethod
    def _classify_value(value: Optional[float], low: Optional[float], high: Optional[float]) -> str:
        if low is None or high is None:
            return "unknown"
        try:
            num_val = float(value) if value is not None else float("nan")
        except Exception:
            return "unknown"

        if num_val != num_val:
            return "unknown"

        if high < low:
            low, high = high, low

        if low <= num_val <= high:
            return "good"

        width = max(1e-6, high - low)
        if num_val < low and (low - num_val) <= 0.1 * width:
            return "warning"
        if num_val > high and (num_val - high) <= 0.1 * width:
            return "warning"

        return "bad"

    @staticmethod
    def _compute_gauge_range(
        low: Optional[float], high: Optional[float], sample_values: List[Optional[float]]
    ) -> Tuple[float, float, float, float]:
        values = [
            float(v)
            for v in sample_values
            if isinstance(v, (int, float)) and v == v  # filter NaN
        ]

        if low is not None and high is not None and high != low:
            if high < low:
                low, high = high, low
            center = (low + high) / 2.0
            span = max(1.0, abs(high - low) * 1.5)
            gmin = round(center - span / 2.0, 2)
            gmax = round(center + span / 2.0, 2)
            return low, high, gmin, gmax

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

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        try:
            val = float(value)
            return val
        except Exception:
            return None

    @staticmethod
    def _epa_calc_tag_names() -> Dict[str, str]:
        return {
            "NOx": "EPA19:NOx_LBHR",
            "CO": "EPA19:CO_LBHR",
            "O2": "EPA19:O2_LBHR",
        }

    def _epa_ppm_to_lbhr_map(self) -> Dict[str, str]:
        calc_tags = self._epa_calc_tag_names()
        mapping: Dict[str, str] = {}
        if self.epa_nox_tag:
            mapping[self.epa_nox_tag] = calc_tags["NOx"]
        if self.epa_co_tag:
            mapping[self.epa_co_tag] = calc_tags["CO"]
        if self.epa_o2_tag:
            mapping[self.epa_o2_tag] = calc_tags["O2"]
        return mapping

    def _get_numeric_from_poll(self, data: Dict[str, Tuple[object, str]], tag: str) -> Optional[float]:
        if not tag:
            return None
        entry = data.get(tag)
        if entry is None:
            return None
        if isinstance(entry, tuple) and len(entry) >= 2:
            val, status = entry[0], entry[1]
            if str(status).lower() != "success":
                return None
        else:
            val = entry
        try:
            return float(val)
        except Exception:
            return None

    def _epa_o2_values(self, raw_o2: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if raw_o2 is None:
            return None, None
        if self.epa_o2_units == "ppmv":
            o2_ppmv = raw_o2
            o2_pct = raw_o2 / 10000.0
        else:
            o2_pct = raw_o2
            o2_ppmv = raw_o2 * 10000.0
        return o2_pct, o2_ppmv

    def _correct_ppmv_for_o2(self, ppmv: float, o2_pct: float) -> Optional[float]:
        if ppmv is None:
            return None
        if o2_pct is None:
            return None
        ref = float(self.epa_ref_o2_pct)
        if not (0.0 <= o2_pct < self.EPA19_STD_O2_PCT):
            return None
        if not (0.0 <= ref < self.EPA19_STD_O2_PCT):
            return None
        return ppmv * (self.EPA19_STD_O2_PCT - ref) / (self.EPA19_STD_O2_PCT - o2_pct)

    def _ppmv_to_lb_hr(self, ppmv: float, flow_scfm: float, molecular_weight: float) -> float:
        return (
            ppmv
            * flow_scfm
            * 60.0
            * molecular_weight
            / (1_000_000.0 * self.EPA19_MOLAR_VOLUME_SCF)
        )

    def _compute_epa_method19(self, data: Dict[str, Tuple[object, str]]) -> Dict[str, Tuple[float, str]]:
        if not self.epa_enabled:
            return {}

        flow = self._get_numeric_from_poll(data, self.epa_flow_tag)
        if flow is None:
            return {}

        raw_o2 = self._get_numeric_from_poll(data, self.epa_o2_tag)
        o2_pct, o2_ppmv = self._epa_o2_values(raw_o2)
        calc_tags = self._epa_calc_tag_names()

        results: Dict[str, Tuple[float, str]] = {}
        pollutant_map = [
            ("NOx", self.epa_nox_tag, True),
            ("CO", self.epa_co_tag, True),
            ("O2", self.epa_o2_tag, False),
        ]

        for pollutant, tag, needs_correction in pollutant_map:
            if not tag:
                continue
            raw_val = self._get_numeric_from_poll(data, tag)
            if raw_val is None:
                continue
            if pollutant == "O2":
                ppmv = o2_ppmv if tag == self.epa_o2_tag else raw_val
            else:
                ppmv = raw_val
            if ppmv is None:
                continue
            if needs_correction:
                if o2_pct is None:
                    continue
                corrected_ppmv = self._correct_ppmv_for_o2(ppmv, o2_pct)
                if corrected_ppmv is None:
                    continue
            else:
                corrected_ppmv = ppmv

            mw = self.EPA19_MOLECULAR_WEIGHTS.get(pollutant)
            if mw is None:
                continue
            lb_hr = self._ppmv_to_lb_hr(corrected_ppmv, flow, mw)
            calc_tag = calc_tags[pollutant]
            results[calc_tag] = (lb_hr, "success")
            self.alias_map.setdefault(calc_tag, f"{pollutant} (lb/hr)")
            self.units_map.setdefault(calc_tag, "lb/hr")

        return results

    # ---------------- settings ----------------

    def _load_settings(self):
        """Load machine, tags (with aliases), and general settings from JSON."""
        if not os.path.exists(self.SETTINGS_FILE):
            return
        try:
            with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load settings: {e}", file=sys.stderr)
            return

        # Keep a snapshot for change detection / logging
        if isinstance(data, dict):
            self._last_settings_snapshot = dict(data)
        else:
            self._last_settings_snapshot = {}

        self.machine_name = data.get("machine_name", "")
        self.machine_name_edit.setText(self.machine_name)

        self.ip_edit.setText(data.get("ip", ""))
        self.interval_spin.setValue(int(data.get("interval", 10)))
        self.chunk_spin.setValue(int(data.get("chunk_size", 20)))

        tags_text = data.get("tags", "")
        self.tags_edit.setPlainText(tags_text)
        # parse aliases from stored tags text
        _, self.alias_map = self.parse_tags_and_aliases(tags_text)

        self.log_dir = data.get("log_dir", self.log_dir)
        self.log_dir_edit.setText(self.log_dir)

        self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")
        self.env_events_path = os.path.join(self.log_dir, "env_events.csv")
        self.system_health_path = os.path.join(self.log_dir, "system_health.json")
        self._ensure_env_events_csv()
        self.thresholds = self._load_thresholds_from_log_dir()

        self.stale_intervals_threshold = int(data.get("stale_intervals", 5))
        self.stale_spin.setValue(self.stale_intervals_threshold)

        self.gap_threshold_minutes = int(data.get("gap_threshold_minutes", 5))
        self.gap_spin.setValue(self.gap_threshold_minutes)

        moving_raw = data.get("moving_tags", "")
        if isinstance(moving_raw, list):
            moving_list = [str(x).strip() for x in moving_raw]
        else:
            moving_list = [x.strip() for x in str(moving_raw).split(",") if x.strip()]
        self.moving_tags = moving_list
        self.moving_tags_edit.setText(", ".join(self.moving_tags))

        self.epa_enabled = bool(data.get("epa_enabled", False))
        self.epa_enabled_checkbox.setChecked(self.epa_enabled)
        self.epa_flow_tag = str(data.get("epa_flow_tag", "") or "")
        self.epa_flow_tag_edit.setText(self.epa_flow_tag)
        self.epa_o2_tag = str(data.get("epa_o2_tag", "") or "")
        self.epa_o2_tag_edit.setText(self.epa_o2_tag)
        self.epa_o2_units = str(data.get("epa_o2_units", "percent") or "percent")
        o2_index = self.epa_o2_units_combo.findData(self.epa_o2_units)
        if o2_index >= 0:
            self.epa_o2_units_combo.setCurrentIndex(o2_index)
        self.epa_ref_o2_pct = float(data.get("epa_ref_o2_pct", self.epa_ref_o2_pct))
        self.epa_ref_o2_spin.setValue(self.epa_ref_o2_pct)
        self.epa_nox_tag = str(data.get("epa_nox_tag", "") or "")
        self.epa_nox_tag_edit.setText(self.epa_nox_tag)
        self.epa_co_tag = str(data.get("epa_co_tag", "") or "")
        self.epa_co_tag_edit.setText(self.epa_co_tag)

        # Machine state settings
        self.machine_state_tag_edit.setText(data.get("machine_state_tag", ""))
        pause_when_down = bool(data.get("pause_when_down", False))
        self.poll_mode_combo.setCurrentIndex(1 if pause_when_down else 0)

        # Heartbeat settings
        self.heartbeat_tag_edit.setText(data.get("heartbeat_tag", ""))
        heartbeat_enabled = bool(data.get("heartbeat_enabled", False))
        self.heartbeat_mode_combo.setCurrentIndex(1 if heartbeat_enabled else 0)

        # Writeback settings
        writeback_enabled = bool(data.get("writeback_enabled", False))
        self.writeback_enabled_checkbox.setChecked(writeback_enabled)
        self.writeback_interval_sec = int(data.get("writeback_interval_sec", 60))
        self.writeback_interval_spin.setValue(self.writeback_interval_sec)
        writeback_mappings = data.get("writeback_mappings", "")
        self.writeback_mappings_edit.setPlainText(str(writeback_mappings))

        self.lockdown_enabled = bool(data.get("lockdown_enabled", False))
        prev_block = self.lockdown_checkbox.blockSignals(True)
        self.lockdown_checkbox.setChecked(self.lockdown_enabled)
        self.lockdown_checkbox.blockSignals(prev_block)

        stored_order = data.get("tag_order", [])
        if isinstance(stored_order, list):
            self.tag_order = stored_order

        if self.machine_name:
            self.setWindowTitle(f"CIP Tag Poller & Dashboard - {self.machine_name}")
            self.machine_label.setText(f"Machine: {self.machine_name}")
        else:
            self.machine_label.setText("Machine: —")

        self._update_config_version_label()
        self._apply_lockdown_state(initial_load=True)

    def _save_settings(self):
        """Persist all tags, aliases (via text), machine, and general settings to JSON."""
        old_data = self._last_settings_snapshot or self._read_settings_file()
        data = {
            "machine_name": self.machine_name_edit.text().strip(),
            "ip": self.ip_edit.text().strip(),
            "interval": self.interval_spin.value(),
            "chunk_size": self.chunk_spin.value(),
            "tags": self.tags_edit.toPlainText(),  # aliases encoded in text
            "log_dir": self.log_dir_edit.text().strip(),
            "tag_order": self.tag_order,
            "machine_state_tag": self.machine_state_tag_edit.text().strip(),
            "pause_when_down": self.poll_mode_combo.currentIndex() == 1,
            "heartbeat_tag": self.heartbeat_tag_edit.text().strip(),
            "heartbeat_enabled": self.heartbeat_mode_combo.currentIndex() == 1,
            "writeback_enabled": self.writeback_enabled_checkbox.isChecked(),
            "writeback_interval_sec": self.writeback_interval_spin.value(),
            "writeback_mappings": self.writeback_mappings_edit.toPlainText(),
            "stale_intervals": self.stale_spin.value(),
            "gap_threshold_minutes": self.gap_spin.value(),
            "moving_tags": [t.strip() for t in self.moving_tags_edit.text().split(",") if t.strip()],
            "epa_enabled": self.epa_enabled_checkbox.isChecked(),
            "epa_flow_tag": self.epa_flow_tag_edit.text().strip(),
            "epa_o2_tag": self.epa_o2_tag_edit.text().strip(),
            "epa_o2_units": self.epa_o2_units_combo.currentData(),
            "epa_ref_o2_pct": self.epa_ref_o2_spin.value(),
            "epa_nox_tag": self.epa_nox_tag_edit.text().strip(),
            "epa_co_tag": self.epa_co_tag_edit.text().strip(),
            "lockdown_enabled": self.lockdown_checkbox.isChecked(),
        }
        self.log_dir = data["log_dir"]
        try:
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._log_settings_changes(old_data, data, "Updated via desktop app")
            self._last_settings_snapshot = dict(data)
            self._update_config_version_label()
        except Exception as e:
            print(f"Failed to save settings: {e}", file=sys.stderr)

    def _read_settings_file(self) -> Dict[str, object]:
        if not os.path.exists(self.SETTINGS_FILE):
            return {}
        try:
            with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _config_change_log_path(self) -> str:
        return os.path.join(self.log_dir, "config_changes.csv")

    def _ensure_config_change_log(self) -> str:
        path = self._config_change_log_path()
        ensure_csv(path, self.CONFIG_CHANGE_HEADERS)
        return path

    @staticmethod
    def _current_user() -> str:
        try:
            user = getpass.getuser()
            return user or "Workstation"
        except Exception:
            return "Workstation"

    def _write_config_change(
        self, field: str, old_value: object, new_value: object, reason: str
    ) -> None:
        path = self._ensure_config_change_log()
        ts = datetime.now().isoformat(timespec="seconds")
        row = [
            ts,
            self._current_user(),
            field,
            "" if old_value is None else str(old_value),
            "" if new_value is None else str(new_value),
            reason,
        ]
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"Failed to log config change: {e}", file=sys.stderr)

    def _log_settings_changes(
        self, old: Dict[str, object], new: Dict[str, object], reason: str
    ) -> None:
        watched_fields = [
            ("tags", "Tags list"),
            ("machine_state_tag", "Machine state tag"),
            ("pause_when_down", "Machine state gating"),
            ("heartbeat_tag", "Heartbeat tag"),
            ("heartbeat_enabled", "Heartbeat mode"),
            ("writeback_enabled", "Writeback enabled"),
            ("writeback_interval_sec", "Writeback interval"),
            ("writeback_mappings", "Writeback mappings"),
            ("epa_enabled", "EPA Method 19 enabled"),
            ("epa_flow_tag", "EPA flow tag"),
            ("epa_o2_tag", "EPA O2 tag"),
            ("epa_o2_units", "EPA O2 units"),
            ("epa_ref_o2_pct", "EPA reference O2"),
            ("epa_nox_tag", "EPA NOx tag"),
            ("epa_co_tag", "EPA CO tag"),
        ]
        for key, label in watched_fields:
            old_val = None if old is None else old.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                self._write_config_change(label, old_val, new_val, reason)

    def _compute_config_version_text(self) -> str:
        if not os.path.exists(self.SETTINGS_FILE):
            return "Config: not found"
        try:
            with open(self.SETTINGS_FILE, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            mdate = datetime.fromtimestamp(os.path.getmtime(self.SETTINGS_FILE)).strftime(
                "%Y-%m-%d"
            )
            return f"Config: v{mdate} ({digest[:8]})"
        except Exception:
            return "Config: unavailable"

    def _update_config_version_label(self):
        self.config_version_label.setText(self._compute_config_version_text())

    def _apply_lockdown_state(self, initial_load: bool = False):
        self.lockdown_enabled = self.lockdown_checkbox.isChecked()
        widgets_to_lock = [
            self.machine_name_edit,
            self.ip_edit,
            self.interval_spin,
            self.chunk_spin,
            self.stale_spin,
            self.gap_spin,
            self.moving_tags_edit,
            self.epa_enabled_checkbox,
            self.epa_flow_tag_edit,
            self.epa_o2_tag_edit,
            self.epa_o2_units_combo,
            self.epa_ref_o2_spin,
            self.epa_nox_tag_edit,
            self.epa_co_tag_edit,
            self.machine_state_tag_edit,
            self.poll_mode_combo,
            self.heartbeat_tag_edit,
            self.heartbeat_mode_combo,
            self.writeback_enabled_checkbox,
            self.writeback_interval_spin,
            self.writeback_mappings_edit,
            self.tags_edit,
            self.log_dir_edit,
            self.browse_btn,
        ]
        running = bool(self.poll_thread and self.poll_thread.isRunning())

        for w in widgets_to_lock:
            w.setEnabled(not self.lockdown_enabled)

        self.start_btn.setEnabled(not self.lockdown_enabled and not running)
        self.stop_btn.setEnabled(not self.lockdown_enabled and running)
        self.calc_btn.setEnabled(not self.lockdown_enabled)
        self.rebuild_btn.setEnabled(not self.lockdown_enabled)

        if not initial_load:
            self._save_settings()
        if self.lockdown_enabled and not running:
            QTimer.singleShot(0, self._auto_start_if_locked)

    def _auto_start_if_locked(self):
        if not self.lockdown_enabled:
            return
        if self.poll_thread and self.poll_thread.isRunning():
            return
        self.log_message(
            "Run lock enabled — automatically starting polling with saved settings."
        )
        self.start_polling()

    # ---------------- polling control ----------------

    def choose_log_dir(self):
        if self.lockdown_enabled:
            self.log_message("Run lock enabled — log directory changes are blocked.")
            return
        path = QFileDialog.getExistingDirectory(
            self, "Select Log Directory", self.log_dir_edit.text().strip() or "."
        )
        if path:
            self.log_dir = path
            self.log_dir_edit.setText(path)
            self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")
            self.env_events_path = os.path.join(self.log_dir, "env_events.csv")
        ensure_csv(
            self.hourly_csv_path,
            ["hour_start", "hour_end", "tag", "avg_value", "avg_lb_hr", "sample_count"],
        )
            self._ensure_env_events_csv()
            self.current_log_date = datetime.now().date()
            self.raw_csv_path = self._ensure_raw_csv_for_date(self.current_log_date)
            self.log_message(f"Log directory set to: {path}")

    def start_polling(self):
        if self.lockdown_enabled and self.poll_thread and self.poll_thread.isRunning():
            return
        if self.poll_thread and self.poll_thread.isRunning():
            QMessageBox.warning(self, "Already running", "Polling is already active.")
            return

        ip = self.ip_edit.text().strip()
        if not ip:
            QMessageBox.warning(self, "Missing IP", "Please enter a PLC IP address.")
            return

        tags_text = self.tags_edit.toPlainText().strip()
        tags, alias_map = self.parse_tags_and_aliases(tags_text)

        self.log_dir = self.log_dir_edit.text().strip()
        ensure_dir(self.log_dir)
        self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")
        self.env_events_path = os.path.join(self.log_dir, "env_events.csv")
        ensure_csv(
            self.hourly_csv_path,
            ["hour_start", "hour_end", "tag", "avg_value", "avg_lb_hr", "sample_count"],
        )

        machine_state_tag = self.machine_state_tag_edit.text().strip()
        heartbeat_tag = self.heartbeat_tag_edit.text().strip()
        heartbeat_enabled = (self.heartbeat_mode_combo.currentIndex() == 1)

        self.stale_intervals_threshold = self.stale_spin.value()
        self.gap_threshold_minutes = self.gap_spin.value()
        self.moving_tags = [
            t.strip() for t in self.moving_tags_edit.text().split(",") if t.strip()
        ]
        self.thresholds = self._load_thresholds_from_log_dir()
        self.units_map.clear()

        self.epa_enabled = self.epa_enabled_checkbox.isChecked()
        self.epa_flow_tag = self.epa_flow_tag_edit.text().strip()
        self.epa_o2_tag = self.epa_o2_tag_edit.text().strip()
        self.epa_o2_units = self.epa_o2_units_combo.currentData() or "percent"
        self.epa_ref_o2_pct = self.epa_ref_o2_spin.value()
        self.epa_nox_tag = self.epa_nox_tag_edit.text().strip()
        self.epa_co_tag = self.epa_co_tag_edit.text().strip()

        epa_tags = [self.epa_flow_tag, self.epa_o2_tag, self.epa_nox_tag, self.epa_co_tag]
        for epa_tag in epa_tags:
            if epa_tag and epa_tag not in tags:
                tags.append(epa_tag)

        if not tags and not machine_state_tag and not heartbeat_enabled:
            QMessageBox.warning(
                self,
                "No tags",
                "Please enter at least one tag to poll, a machine state tag, "
                "or enable a heartbeat tag.",
            )
            return

        if len(tags) > 200:
            QMessageBox.warning(
                self,
                "Many tags",
                f"You've configured {len(tags)} tags. The app can handle this, "
                "but consider keeping it to a few dozen for better responsiveness.",
            )

        if self.epa_enabled:
            if not self.epa_flow_tag:
                self.log_message(
                    "EPA Method 19 is enabled, but no air flow tag is configured."
                )
            if not self.epa_o2_tag:
                self.log_message(
                    "EPA Method 19 is enabled, but no O2 correction tag is configured."
                )
            if not (self.epa_nox_tag or self.epa_co_tag or self.epa_o2_tag):
                self.log_message(
                    "EPA Method 19 is enabled, but no NOx/CO/O2 pollutant tags are configured."
                )

        # stable ordering for the dashboard
        tag_order = list(tags)
        if machine_state_tag and machine_state_tag not in tag_order:
            tag_order.append(machine_state_tag)
        if heartbeat_tag and heartbeat_tag not in tag_order:
            tag_order.append(heartbeat_tag)
        self.tag_order = tag_order
        self.alias_map = alias_map  # keep for CSV + table

        if self.epa_enabled:
            calc_tags = self._epa_calc_tag_names()
            for pollutant, calc_tag in calc_tags.items():
                if pollutant == "NOx" and not self.epa_nox_tag:
                    continue
                if pollutant == "CO" and not self.epa_co_tag:
                    continue
                if pollutant == "O2" and not self.epa_o2_tag:
                    continue
                if calc_tag not in self.tag_order:
                    self.tag_order.append(calc_tag)
                self.alias_map.setdefault(calc_tag, f"{pollutant} (lb/hr)")
                self.units_map.setdefault(calc_tag, "lb/hr")

        # machine name / header
        self.machine_name = self.machine_name_edit.text().strip()
        if self.machine_name:
            self.setWindowTitle(f"CIP Tag Poller & Dashboard - {self.machine_name}")
            self.machine_label.setText(f"Machine: {self.machine_name}")
        else:
            self.setWindowTitle("CIP Tag Poller & Dashboard")
            self.machine_label.setText("Machine: —")

        # reset per-run state
        self.current_values.clear()
        self.current_hour_preview.clear()
        self.current_hour_start = None
        self.hour_accumulators.clear()
        self.last_success_ts.clear()
        self._stale_tracker.clear()
        self.update_dashboard()

        interval = self.interval_spin.value()
        chunk_size = self.chunk_spin.value()
        pause_when_down = (self.poll_mode_combo.currentIndex() == 1)

        # ensure current day's raw file exists before starting
        self.current_log_date = datetime.now().date()
        self.raw_csv_path = self._ensure_raw_csv_for_date(self.current_log_date)
        self._ensure_env_events_csv()

        self.poll_thread = PollThread(
            ip=ip,
            tags=tags,
            interval_sec=interval,
            chunk_size=chunk_size,
            reconnect_delay_sec=10,
            machine_state_tag=machine_state_tag,
            pause_when_down=pause_when_down,
            heartbeat_tag=heartbeat_tag,
            heartbeat_enabled=heartbeat_enabled,
            parent=self,
        )
        self.poll_thread.data_ready.connect(self.handle_poll_data)
        self.poll_thread.error.connect(self.handle_poll_error)
        self.poll_thread.info.connect(self.log_message)
        self.poll_thread.start()

        self._start_writeback_thread(ip)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(not self.lockdown_enabled)
        self.connection_status_label.setText("Status: Polling")
        self.connection_status_label.setStyleSheet("color: #a5d6a7;")

        mode_txt = "Pause when machine down" if pause_when_down else "Poll always"
        hb_txt = "enabled" if heartbeat_enabled else "disabled"
        self.log_message(
            f"Started polling every {interval} seconds for {len(tags)} tag(s) "
            f"(mode={mode_txt}, heartbeat {hb_txt})."
        )

    def stop_polling(self):
        if self.lockdown_enabled:
            self.log_message("Run lock enabled — stop is blocked.")
            return
        if self.poll_thread:
            self.poll_thread.stop()
            self.poll_thread.wait(5000)
            self.poll_thread = None
        if self.writeback_thread:
            self.writeback_thread.stop()
            self.writeback_thread.wait(5000)
            self.writeback_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.connection_status_label.setText("Status: Stopped")
        self.connection_status_label.setStyleSheet("color: #ef9a9a;")
        self.log_message("Stopped polling.")

    def _start_writeback_thread(self, ip: str) -> None:
        if not self.writeback_enabled_checkbox.isChecked():
            return

        mappings = self.parse_writeback_mappings(
            self.writeback_mappings_edit.toPlainText()
        )
        resolved = self._resolve_writeback_targets(mappings)
        if not resolved:
            self.log_message(
                "Writeback enabled but no valid alias mappings were found; skipping."
            )
            return

        interval_sec = self.writeback_interval_spin.value()
        self.writeback_thread = WritebackThread(
            ip=ip,
            interval_sec=interval_sec,
            writeback_map=resolved,
            parent=self,
        )
        self.writeback_thread.info.connect(self.log_message)
        self.writeback_thread.error.connect(self.handle_poll_error)
        self.writeback_thread.update_averages(self._current_hour_averages())
        self.writeback_thread.start()

    # ---------------- data handling ----------------

    def handle_poll_error(self, msg: str):
        """Handle errors emitted from PollThread (no modal boxes)."""
        self.log_message(msg)
        self._update_system_health(error=True)

    def _update_stale_state(self, tag: str, value: object) -> bool:
        """Return True if tag is considered stale based on repeated identical values."""
        tracker = self._stale_tracker.setdefault(tag, {"last": None, "count": 0})
        last_val = tracker.get("last")
        if last_val is None:
            tracker["count"] = 1
        elif last_val == value:
            tracker["count"] = int(tracker.get("count", 0)) + 1
        else:
            tracker["count"] = 1
        tracker["last"] = value
        return tracker.get("count", 0) >= self.stale_intervals_threshold

    def _check_gap_event(self, tag: str, ts: datetime) -> None:
        """Log a DATA_GAP event if the time since the last successful sample exceeds threshold."""
        last_ts = self.last_success_ts.get(tag)
        self.last_success_ts[tag] = ts
        if not last_ts:
            return
        delta_sec = (ts - last_ts).total_seconds()
        if delta_sec <= self.gap_threshold_minutes * 60:
            return
        try:
            self._ensure_env_events_csv()
            with open(self.env_events_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ts.isoformat(timespec="seconds"), "DATA_GAP", tag, int(delta_sec)])
        except Exception as e:
            self.log_message(f"Error logging DATA_GAP event: {e}")

    def _determine_qa_flag(self, tag: str, val: object, status: str) -> str:
        status_lower = str(status).lower()
        if status_lower != "success":
            return "PLC_ERROR"
        if val is None:
            return "MISSING"
        try:
            num_val = float(val)
        except Exception:
            return "SUSPECT"

        if num_val != num_val or num_val in (float("inf"), float("-inf")):
            return "SUSPECT"

        thresholds = self.thresholds.get(tag)
        if isinstance(thresholds, dict):
            try:
                low_limit = _to_float(thresholds.get("low_limit"))
                high_limit = _to_float(thresholds.get("high_limit"))
                low_oper = _to_float(thresholds.get("low_oper"))
                high_oper = _to_float(thresholds.get("high_oper"))

                low_bound = low_limit if low_limit is not None else low_oper
                high_bound = high_limit if high_limit is not None else high_oper

                if low_bound is not None and num_val < low_bound:
                    return "OUT_OF_RANGE"
                if high_bound is not None and num_val > high_bound:
                    return "OUT_OF_RANGE"
            except Exception:
                pass

        if tag in self.moving_tags:
            if self._update_stale_state(tag, num_val):
                return "STALE"

        if "manual" in status_lower:
            return "MANUAL_CORRECTION"

        return "OK"

    def handle_poll_data(self, ts_iso: str, data: Dict[str, Tuple[object, str]]):
        """
        Called from background thread with new readings.
        - Update current values
        - Append to daily raw CSV (raw_data_YYYY-MM-DD.csv)
        - Update hourly accumulators and roll over when hour changes
        """
        try:
            data = dict(data)
            epa_results = self._compute_epa_method19(data)
            if epa_results:
                for calc_tag in epa_results:
                    if calc_tag not in self.tag_order:
                        self.tag_order.append(calc_tag)
                data.update(epa_results)

            try:
                ts = datetime.fromisoformat(ts_iso)
            except Exception:
                ts = datetime.now()

            hour_start = hour_bucket(ts)

            if self.current_hour_start is None:
                self.current_hour_start = hour_start
            elif hour_start != self.current_hour_start:
                self.finalize_previous_hour()
                self.current_hour_start = hour_start
                self.hour_accumulators.clear()
                self.current_hour_preview.clear()

            # Daily rotation: if date changed, switch raw CSV
            log_date = ts.date()
            if log_date != self.current_log_date or not self.raw_csv_path:
                self.current_log_date = log_date
                self.raw_csv_path = self._ensure_raw_csv_for_date(self.current_log_date)
            else:
                ensure_csv(
                    self.raw_csv_path,
                    ["timestamp", "date", "time", "tag", "alias", "value", "status", "qa_flag"],
                )

            date_str = ts.strftime("%Y-%m-%d")
            time_str = ts.strftime("%H:%M:%S")

            try:
                with open(self.raw_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for tag, (val, status) in data.items():
                        qa_flag = self._determine_qa_flag(tag, val, status)
                        self.current_values[tag] = (val, qa_flag)
                        alias = self.alias_map.get(tag, "")
                        writer.writerow(
                            [ts_iso, date_str, time_str, tag, alias, val, status, qa_flag]
                        )

                        if qa_flag == "OK":
                            self._check_gap_event(tag, ts)
                            if isinstance(val, (int, float)):
                                acc = self.hour_accumulators.setdefault(
                                    tag, {"sum": 0.0, "count": 0}
                                )
                                acc["sum"] += float(val)
                                acc["count"] += 1
            except Exception as e:
                self.log_message(f"Error writing raw CSV: {e}")

            if self.writeback_thread and self.writeback_thread.isRunning():
                self.writeback_thread.update_averages(self._current_hour_averages())

            self.last_update_label.setText(f"Last update: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            self._update_system_health(success_ts=ts)
            self.update_dashboard()
        except Exception as e:
            self.log_message(f"Error handling poll data: {e}")

    def _current_hour_averages(self) -> Dict[str, float]:
        averages: Dict[str, float] = {}
        for tag, acc in self.hour_accumulators.items():
            if acc.get("count", 0) > 0:
                averages[tag] = acc["sum"] / acc["count"]
        return averages

    def finalize_previous_hour(self):
        """Compute averages for tags in the previous hour and write to hourly CSV."""
        if self.current_hour_start is None or not self.hour_accumulators:
            return

        hour_start = self.current_hour_start
        hour_end = hour_start + timedelta(hours=1)
        hour_start_iso = hour_start.isoformat(timespec="seconds")
        hour_end_iso = hour_end.isoformat(timespec="seconds")

        records: Dict[Tuple[str, str], Tuple[str, Optional[float], Optional[float], int]] = {}
        try:
            if os.path.exists(self.hourly_csv_path):
                with open(self.hourly_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = (row["hour_start"], row["tag"])
                        records[key] = (
                            row["hour_end"],
                            self._safe_float(row.get("avg_value")),
                            self._safe_float(row.get("avg_lb_hr")),
                            int(row.get("sample_count", "0") or 0),
                        )
        except Exception as e:
            self.log_message(f"Error reading hourly CSV for upsert: {e}")

        ppm_to_lbhr = self._epa_ppm_to_lbhr_map()
        lbhr_avgs: Dict[str, float] = {}
        for tag, acc in self.hour_accumulators.items():
            if acc["count"] <= 0:
                continue
            if isinstance(acc["sum"], (int, float)):
                lbhr_avgs[tag] = acc["sum"] / acc["count"]

        self.last_hour_avg.clear()
        for tag, acc in self.hour_accumulators.items():
            if acc["count"] <= 0:
                continue
            avg = acc["sum"] / acc["count"]
            self.last_hour_avg[tag] = avg
            key = (hour_start_iso, tag)
            existing = records.get(key)
            existing_lb_hr = existing[2] if existing else None
            lbhr_avg = existing_lb_hr
            mapped_tag = ppm_to_lbhr.get(tag)
            if mapped_tag:
                lbhr_avg = lbhr_avgs.get(mapped_tag, existing_lb_hr)
            records[key] = (hour_end_iso, avg, lbhr_avg, int(acc["count"]))

        try:
            ensure_csv(
                self.hourly_csv_path,
                ["hour_start", "hour_end", "tag", "avg_value", "avg_lb_hr", "sample_count"],
            )
            with open(self.hourly_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["hour_start", "hour_end", "tag", "avg_value", "avg_lb_hr", "sample_count"]
                )
                for (hs, tag), (he, avg, lbhr_avg, cnt) in sorted(records.items()):
                    writer.writerow(
                        [
                            hs,
                            he,
                            tag,
                            "" if avg is None else avg,
                            "" if lbhr_avg is None else lbhr_avg,
                            cnt,
                        ]
                    )
            self.log_message(
                f"Hourly averages upserted for hour starting {hour_start_iso}."
            )
        except Exception as e:
            self.log_message(f"Error writing hourly CSV: {e}")

    def compute_current_hour_preview(self):
        self.current_hour_preview.clear()
        for tag, acc in self.hour_accumulators.items():
            if acc["count"] <= 0:
                continue
            self.current_hour_preview[tag] = acc["sum"] / acc["count"]
        self.update_dashboard()
        self.log_message("Computed current hour averages (display only).")

    def rebuild_hourly_from_raw(self):
        """
        Recompute all hourly averages from all raw_data_YYYY-MM-DD.csv
        files in the log directory and rewrite hourly_averages.csv.
        """
        if not os.path.isdir(self.log_dir):
            QMessageBox.warning(
                self, "No Raw Data", "Log directory does not exist; nothing to rebuild."
            )
            return

        raw_files = []
        try:
            for name in os.listdir(self.log_dir):
                if not name.startswith("raw_data_"):
                    continue
                if name.endswith(".csv") or name.endswith(".csv.gz"):
                    raw_files.append(os.path.join(self.log_dir, name))
        except Exception as e:
            self.log_message(f"Error listing raw files: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to list raw data files:\n{e}"
            )
            return

        if not raw_files:
            QMessageBox.warning(
                self,
                "No Raw Data",
                "No raw_data_YYYY-MM-DD.csv files found; nothing to rebuild.",
            )
            return

        self.log_message(
            f"Rebuilding hourly averages from {len(raw_files)} raw data file(s)..."
        )
        records: Dict[Tuple[str, str], Tuple[str, Optional[float], Optional[float], int]] = {}

        try:
            buckets: Dict[Tuple[datetime, str], Dict[str, float]] = {}
            for path in sorted(raw_files):
                open_func = gzip.open if path.endswith(".gz") else open
                with open_func(path, "rt", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ts_str = row.get("timestamp", "")
                        if not ts_str:
                            continue
                        try:
                            ts = datetime.fromisoformat(ts_str)
                        except Exception:
                            continue

                        tag = row.get("tag", "")
                        if not tag:
                            continue
                        status = str(row.get("status", "")).lower()
                        qa_flag = str(row.get("qa_flag", "")).upper()
                        try:
                            val = float(row.get("value", ""))
                        except Exception:
                            continue
                        if qa_flag and qa_flag != "OK":
                            continue
                        if status != "success":
                            continue

                        hstart = hour_bucket(ts)
                        key = (hstart, tag)
                        acc = buckets.setdefault(key, {"sum": 0.0, "count": 0})
                        acc["sum"] += val
                        acc["count"] += 1

            for (hstart, tag), acc in buckets.items():
                if acc["count"] <= 0:
                    continue
                hstart_iso = hstart.isoformat(timespec="seconds")
                hend_iso = (hstart + timedelta(hours=1)).isoformat(timespec="seconds")
                avg = acc["sum"] / acc["count"]
                records[(hstart_iso, tag)] = (hend_iso, avg, None, int(acc["count"]))
        except Exception as e:
            self.log_message(f"Error rebuilding hourly from raw: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to rebuild hourly averages:\n{e}"
            )
            return

        ppm_to_lbhr = self._epa_ppm_to_lbhr_map()
        if ppm_to_lbhr:
            for (hstart_iso, tag), (hend_iso, avg, lbhr_avg, count) in list(records.items()):
                lbhr_tag = ppm_to_lbhr.get(tag)
                if not lbhr_tag:
                    continue
                lbhr_record = records.get((hstart_iso, lbhr_tag))
                if lbhr_record:
                    records[(hstart_iso, tag)] = (
                        hend_iso,
                        avg,
                        lbhr_record[1],
                        count,
                    )

        try:
            ensure_csv(
                self.hourly_csv_path,
                ["hour_start", "hour_end", "tag", "avg_value", "avg_lb_hr", "sample_count"],
            )
            with open(self.hourly_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["hour_start", "hour_end", "tag", "avg_value", "avg_lb_hr", "sample_count"]
                )
                for (hs, tag), (he, avg, lbhr_avg, cnt) in sorted(records.items()):
                    writer.writerow(
                        [
                            hs,
                            he,
                            tag,
                            "" if avg is None else avg,
                            "" if lbhr_avg is None else lbhr_avg,
                            cnt,
                        ]
                    )
            self.log_message(
                f"Rebuilt hourly averages for {len(records)} (hour, tag) pairs."
            )
        except Exception as e:
            self.log_message(f"Error writing rebuilt hourly CSV: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to write rebuilt hourly CSV:\n{e}"
            )

    # ---------------- dashboard rendering ----------------

    def update_dashboard(self):
        """
        Render table rows:
            Alias | Tag | Current Value | Last Hour Avg | Current Hour Avg (so far)
            | Last Hour Avg (lb/hr) | Current Hour Avg (lb/hr) | QA Flag
        """
        if self.tag_order:
            tags = list(self.tag_order)
        else:
            tags = sorted(
                set(self.current_values.keys())
                | set(self.last_hour_avg.keys())
                | set(self.current_hour_preview.keys())
            )

        ppm_to_lbhr = self._epa_ppm_to_lbhr_map()
        self.table.setRowCount(len(tags))
        self.rows_summary_label.setText(f"Rows: {len(tags)}")

        for row, tag in enumerate(tags):
            entry = self.current_values.get(tag, ("", ""))
            if isinstance(entry, tuple) and len(entry) >= 2:
                val, qa_flag = entry[0], entry[1]
            else:
                val, qa_flag = entry, ""

            last_avg = self.last_hour_avg.get(tag, "")
            cur_avg = self.current_hour_preview.get(tag, "")
            lbhr_tag = ppm_to_lbhr.get(tag)
            lbhr_last_avg = self.last_hour_avg.get(lbhr_tag, "") if lbhr_tag else ""
            lbhr_cur_avg = self.current_hour_preview.get(lbhr_tag, "") if lbhr_tag else ""

            alias = self.alias_map.get(tag, "")

            def item(text):
                it = QTableWidgetItem(str(text))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                return it

            self.table.setItem(row, 0, item(alias))
            self.table.setItem(row, 1, item(tag))
            self.table.setItem(row, 2, item(val))
            self.table.setItem(
                row,
                3,
                item(f"{last_avg:.4f}" if isinstance(last_avg, (int, float)) else ""),
            )
            self.table.setItem(
                row,
                4,
                item(f"{cur_avg:.4f}" if isinstance(cur_avg, (int, float)) else ""),
            )
            self.table.setItem(
                row,
                5,
                item(f"{lbhr_last_avg:.4f}" if isinstance(lbhr_last_avg, (int, float)) else ""),
            )
            self.table.setItem(
                row,
                6,
                item(f"{lbhr_cur_avg:.4f}" if isinstance(lbhr_cur_avg, (int, float)) else ""),
            )
            self.table.setItem(row, 7, item(qa_flag))

        self._update_gauges(tags)

    def _update_gauges(self, tags: List[str]) -> None:
        if not hasattr(self, "gauge_list_layout"):
            return

        if self._gauge_spacer is not None:
            self.gauge_list_layout.removeItem(self._gauge_spacer)

        active_tags: List[str] = []
        for tag in tags:
            card = self.gauge_cards.get(tag)
            if card is None:
                card = GaugeCard(tag)
                self.gauge_cards[tag] = card
            else:
                self.gauge_list_layout.removeWidget(card)

            thresholds_entry = self.thresholds.get(tag, {}) if isinstance(self.thresholds.get(tag, {}), dict) else {}
            alias = thresholds_entry.get("alias") if isinstance(thresholds_entry.get("alias"), str) else None
            units = thresholds_entry.get("units") if isinstance(thresholds_entry.get("units"), str) else ""
            if not units and tag in self.units_map:
                units = self.units_map.get(tag, "")

            low_oper = _to_float(thresholds_entry.get("low_oper"))
            high_oper = _to_float(thresholds_entry.get("high_oper"))
            low_limit = _to_float(thresholds_entry.get("low_limit"))
            high_limit = _to_float(thresholds_entry.get("high_limit"))

            alias_display = alias or self.alias_map.get(tag, "") or tag
            card.update_metadata(alias_display, tag, units or "", low_oper, high_oper, low_limit, high_limit)

            entry = self.current_values.get(tag, (None, ""))
            if isinstance(entry, tuple) and len(entry) >= 2:
                current_val_raw, qa_flag = entry[0], entry[1]
            else:
                current_val_raw, qa_flag = entry, ""

            current_val = self._safe_float(current_val_raw)
            last_avg = self._safe_float(self.last_hour_avg.get(tag))
            live_avg = self._safe_float(self.current_hour_preview.get(tag))

            sample_vals: List[Optional[float]] = [current_val, last_avg, live_avg]
            low_for_range = low_oper if low_oper is not None else low_limit
            high_for_range = high_oper if high_oper is not None else high_limit
            low_eff, high_eff, gauge_min, gauge_max = self._compute_gauge_range(
                low_for_range, high_for_range, sample_vals
            )
            low_for_class = low_oper if low_oper is not None else low_eff
            high_for_class = high_oper if high_oper is not None else high_eff

            cur_status = self._classify_value(current_val, low_for_class, high_for_class)
            last_status = self._classify_value(last_avg, low_for_class, high_for_class)
            live_status = self._classify_value(live_avg, low_for_class, high_for_class)

            status_bits = []
            if cur_status != "unknown":
                status_bits.append(f"Status: {cur_status}")
            if qa_flag:
                status_bits.append(f"QA: {qa_flag}")
            current_detail = " | ".join(status_bits) if status_bits else "Status: unknown"
            last_detail = "Status: unknown" if last_status == "unknown" else f"Status: {last_status}"
            live_detail = "Status: unknown" if live_status == "unknown" else f"Status: {live_status}"

            card.update_values(
                current_val,
                last_avg,
                live_avg,
                gauge_min,
                gauge_max,
                self._status_color(cur_status),
                self._status_color(last_status),
                self._status_color(live_status),
                current_detail,
                last_detail,
                live_detail,
            )

            self.gauge_list_layout.addWidget(card)
            active_tags.append(tag)

        stale_tags = set(self.gauge_cards.keys()) - set(active_tags)
        for tag in stale_tags:
            widget = self.gauge_cards.pop(tag)
            widget.setParent(None)
            widget.deleteLater()

        self._gauge_spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.gauge_list_layout.addItem(self._gauge_spacer)

    # ---------------- logging & lifecycle ----------------

    def log_message(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_edit.appendPlainText(line)
        self.statusBar().showMessage(msg, 5000)

    def closeEvent(self, event):
        if self.lockdown_enabled:
            QMessageBox.warning(
                self,
                "Run lock enabled",
                "The application cannot be closed while run lock is active.",
            )
            event.ignore()
            return
        if self.poll_thread and self.poll_thread.isRunning():
            self.poll_thread.stop()
            self.poll_thread.wait(5000)
        if self.writeback_thread and self.writeback_thread.isRunning():
            self.writeback_thread.stop()
            self.writeback_thread.wait(5000)
        self._save_settings()
        event.accept()


# ------------------- launch helpers -------------------


def _build_splash_pixmap(icon: QIcon) -> QPixmap:
    """Create a splash pixmap using the app icon when available."""
    screen = QApplication.primaryScreen()
    target_size = 360
    if screen:
        available = screen.availableGeometry()
        target_size = int(min(available.width(), available.height()) * 0.35)
        target_size = max(240, min(target_size, 512))

    if not icon.isNull():
        pixmap = icon.pixmap(target_size, target_size)
    else:
        pixmap = QPixmap()

    if pixmap.isNull():
        pixmap = QPixmap(target_size, target_size)
        pixmap.fill(QColor("#1c252c"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor("#9bd6ff"))
        painter.setFont(QFont("Arial", int(target_size / 8), QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "CIP Tag\nPoller")
        painter.end()

    return pixmap


# ------------------- main entry -------------------


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QGuiApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QApplication(sys.argv)

    icon_path = resource_path("ram.ico")
    app_icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    splash_pixmap = _build_splash_pixmap(app_icon)
    splash = QSplashScreen(splash_pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    win = MainWindow()
    if not app_icon.isNull():
        win.setWindowIcon(app_icon)
    win.showMaximized()

    splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - defensive startup guard
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup_error.log")
        tb_text = traceback.format_exc()
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {exc}\n\n{tb_text}")
        except Exception:
            log_path = None

        message_lines = [
            "CIP Tag Poller failed to start.",
            f"Error: {exc}",
        ]
        if log_path:
            message_lines.append(f"Details were written to {log_path}.")

        print("\n".join(message_lines), file=sys.stderr)

        try:
            app = QApplication.instance()
            owns_app = False
            if app is None:
                app = QApplication(sys.argv)
                owns_app = True
            QMessageBox.critical(None, "CIP startup error", "\n".join(message_lines))
            if owns_app:
                app.quit()
        except Exception:
            pass

        sys.exit(1)
