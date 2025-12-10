#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIP Tag Poller & Dashboard (PySide6, dark themed, production-ready)

Features
--------
- Polls a list of PLC tags at a configurable interval (default 10s) using pylogix.
- Logs all raw samples to logs/raw_data.csv (timestamp, tag, value, status).
- Aggregates data into hourly averages (1–2pm, 2–3pm, …) and writes to
  logs/hourly_averages.csv with UPSERT semantics:
  for each (hour_start, tag) there is at most one row (latest wins).
- Dark, professional GUI showing:
    Tag | Current Value | Last Hour Avg | Current Hour Avg (so far) | Status
- Button to compute current hour averages (display only).
- Button to recompute/rebuild hourly_averages.csv from raw_data.csv.
- Settings (machine, IP, tags, interval, chunk size, log dir) are remembered
  in settings.json so we can relaunch fast.

Reliability / Always-on
-----------------------
- PollThread uses a reconnect loop: if PLC connect fails, it logs the error and
  retries after a delay, indefinitely, until you press Stop.
- Read errors are caught and logged, not fatal.
- If a multi-read hits "too many parameters", the thread automatically falls
  back to single-tag reads for that batch.
- No modal message boxes from the background thread, to avoid freezes.
"""

import sys
import os
import csv
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

from PySide6.QtCore import Qt, QThread, Signal, QMutex, QMutexLocker
from PySide6.QtGui import QPalette, QColor, QAction, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QPushButton, QTextEdit, QTableWidget,
    QTableWidgetItem, QMessageBox, QGroupBox, QFormLayout, QSpinBox,
    QFileDialog, QHeaderView, QPlainTextEdit, QStatusBar, QFrame,
    QSpacerItem, QSizePolicy
)

try:
    from pylogix import PLC  # type: ignore
    PYLOGIX_AVAILABLE = True
except Exception:  # ImportError or others
    PYLOGIX_AVAILABLE = False


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


def chunked(seq: List[str], size: int):
    """Yield chunks from list (used to batch pylogix reads)."""
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


def hour_bucket(dt: datetime) -> datetime:
    """Return datetime truncated to hour (start of hour)."""
    return dt.replace(minute=0, second=0, microsecond=0)


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
    """

    data_ready = Signal(str, dict)
    error = Signal(str)
    info = Signal(str)

    # Conservative upper bound for CIP multi-read requests.
    # 10 tags per read is very safe for typical ControlLogix/CompactLogix setups.
    MAX_SAFE_CHUNK = 10

    def __init__(
        self,
        ip: str,
        tags: List[str],
        interval_sec: int = 10,
        chunk_size: int = 20,
        reconnect_delay_sec: int = 10,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.ip = ip
        # keep the raw order; this will be used in the table as well
        self.tags = [t for t in tags if t.strip()]
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

        # How long to wait before reattempting a PLC connection after failure
        self.reconnect_delay_sec = max(1, int(reconnect_delay_sec))

        self._running = False
        self._mutex = QMutex()

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
                # PLC rejected the multi-read; fall back to single-tag reads
                self.info.emit(
                    "PLC reported 'too many parameters' on multi-read; "
                    "falling back to single-tag reads for this batch."
                )
            else:
                # Other read error; re-raise to be handled by caller
                raise
        else:
            # Multi-read succeeded; capture results
            for r in res:
                tag = getattr(r, "TagName", "(unknown)")
                val = getattr(r, "Value", None)
                st = getattr(r, "Status", "(unknown)")
                results[tag] = (val, st)
            return results

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
                # mark as failed but keep going
                results[tag] = (None, f"Error: {e}")
        return results

    # ---------- main thread loop ----------

    def run(self):
        if not PYLOGIX_AVAILABLE:
            self.error.emit("pylogix is not installed. Run: pip install pylogix")
            return

        if not self.tags:
            self.error.emit("No tags specified to poll.")
            return

        with QMutexLocker(self._mutex):
            self._running = True

        # Log any chunk_size clamping once we have signals active
        if getattr(self, "_clamped_chunk_msg", ""):
            self.info.emit(self._clamped_chunk_msg)

        # Always-on reconnect loop
        while self._should_run():
            try:
                with PLC() as comm:
                    comm.IPAddress = self.ip
                    self.info.emit(
                        f"Connected to PLC at {self.ip} "
                        f"({len(self.tags)} tags, chunk_size={self.chunk_size})"
                    )

                    # Inner polling loop while connection is alive
                    while self._should_run():
                        start_t = time.time()
                        all_results: Dict[str, Tuple[object, str]] = {}

                        try:
                            # batched multi-reads so many tags are handled efficiently,
                            # with automatic fallback if PLC complains.
                            for batch in chunked(self.tags, self.chunk_size):
                                if not batch:
                                    continue
                                batch_results = self._multi_read_with_fallback(comm, batch)
                                all_results.update(batch_results)
                        except Exception as e:
                            # per-cycle read error; log and retry next cycle
                            self.error.emit(f"Read error: {e}")
                            time.sleep(self.interval_sec)
                            continue

                        ts_iso = datetime.now().isoformat(timespec="seconds")
                        self.data_ready.emit(ts_iso, all_results)

                        elapsed = time.time() - start_t
                        sleep_for = max(0.0, self.interval_sec - elapsed)
                        # Sleep in small slices so stop() can interrupt quickly
                        end_time = time.time() + sleep_for
                        while time.time() < end_time:
                            if not self._should_run():
                                break
                            time.sleep(0.1)

            except Exception as e:
                # Connection-level failure: log and retry after a delay
                self.error.emit(f"PLC connection error: {e}")
                # Wait reconnect_delay_sec, but stay responsive to stop()
                end_time = time.time() + self.reconnect_delay_sec
                while time.time() < end_time:
                    if not self._should_run():
                        break
                    time.sleep(0.5)

        self.info.emit("Polling thread stopped.")


# ------------------- main window -------------------


class MainWindow(QMainWindow):
    SETTINGS_FILE = "settings.json"

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CIP Tag Poller & Dashboard")
        self.resize(1300, 750)

        # --- state ---
        self.poll_thread: Optional[PollThread] = None
        self.current_values: Dict[str, Tuple[object, str]] = {}
        self.last_hour_avg: Dict[str, float] = {}
        self.current_hour_preview: Dict[str, float] = {}
        self.current_hour_start: Optional[datetime] = None
        self.hour_accumulators: Dict[str, Dict[str, float]] = {}

        # stable tag ordering for the dashboard
        self.tag_order: List[str] = []

        # default config
        self.machine_name = ""
        self.log_dir = os.path.join("logs")
        self.raw_csv_path = os.path.join(self.log_dir, "raw_data.csv")
        self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")

        # UI
        self._build_ui()
        self._apply_dark_theme()
        self._load_settings()  # populates fields and paths

        # ensure CSVs
        ensure_csv(self.raw_csv_path, ["timestamp", "tag", "value", "status"])
        ensure_csv(
            self.hourly_csv_path,
            ["hour_start", "hour_end", "tag", "avg_value", "sample_count"],
        )

    # ---------------- UI construction ----------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

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

        header_left = QVBoxLayout()
        header_left.addWidget(self.title_label)
        header_left.addWidget(self.machine_label)

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

        # Connection / tags group (styled as a simple vertical layout)
        cfg_group = QGroupBox("Configuration")
        cfg_group.setObjectName("cardGroupBox")
        cfg_layout = QFormLayout(cfg_group)
        cfg_layout.setLabelAlignment(Qt.AlignLeft)
        cfg_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        cfg_layout.setHorizontalSpacing(10)
        cfg_layout.setVerticalSpacing(6)

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

        tags_label = QLabel("Tags to Poll (one per line):")
        self.tags_edit = QTextEdit()
        self.tags_edit.setFixedHeight(160)
        self.tags_edit.setPlaceholderText(
            "e.g.\nProgram:MainRoutine.MyTag1\nProgram:MainRoutine.MyTag2\n..."
        )
        cfg_layout.addRow(tags_label, self.tags_edit)

        # Log directory selector
        dir_layout = QHBoxLayout()
        self.log_dir_edit = QLineEdit(self.log_dir)
        browse_btn = QPushButton("Browse…")
        browse_btn.setObjectName("secondaryButton")
        dir_layout.addWidget(self.log_dir_edit)
        dir_layout.addWidget(browse_btn)
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

        # Bottom spacer to keep things tight at top
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

        # Dashboard header
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

        right_layout.addLayout(dash_header_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            [
                "Tag",
                "Current Value",
                "Last Hour Avg",
                "Current Hour Avg (so far)",
                "Status",
            ]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        right_layout.addWidget(self.table, stretch=1)

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
        browse_btn.clicked.connect(self.choose_log_dir)

        # Menu (exit)
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
        palette.setColor(QPalette.Highlight, QColor(76, 175, 80))  # green accent
        palette.setColor(QPalette.HighlightedText, Qt.white)

        app.setPalette(palette)

        # Global stylesheet – VisionFinal-like, card-based, flat buttons
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
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox {
                background-color: #181c22;
                border: 1px solid #2b323c;
                border-radius: 4px;
                padding: 4px 6px;
                color: #e0e6ed;
                selection-background-color: #4caf50;
            }
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus {
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

    # ---------------- settings ----------------

    def _load_settings(self):
        """Load machine, tags, and general settings from JSON."""
        if not os.path.exists(self.SETTINGS_FILE):
            return
        try:
            with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load settings: {e}", file=sys.stderr)
            return

        self.machine_name = data.get("machine_name", "")
        self.machine_name_edit.setText(self.machine_name)

        self.ip_edit.setText(data.get("ip", ""))
        self.interval_spin.setValue(int(data.get("interval", 10)))
        self.chunk_spin.setValue(int(data.get("chunk_size", 20)))

        self.tags_edit.setPlainText(data.get("tags", ""))

        self.log_dir = data.get("log_dir", self.log_dir)
        self.log_dir_edit.setText(self.log_dir)
        self.raw_csv_path = os.path.join(self.log_dir, "raw_data.csv")
        self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")

        stored_order = data.get("tag_order", [])
        if isinstance(stored_order, list):
            self.tag_order = stored_order

        if self.machine_name:
            self.setWindowTitle(f"CIP Tag Poller & Dashboard - {self.machine_name}")
            self.machine_label.setText(f"Machine: {self.machine_name}")
        else:
            self.machine_label.setText("Machine: —")

    def _save_settings(self):
        """Persist all tags, machine, and general settings to JSON."""
        data = {
            "machine_name": self.machine_name_edit.text().strip(),
            "ip": self.ip_edit.text().strip(),
            "interval": self.interval_spin.value(),
            "chunk_size": self.chunk_spin.value(),
            "tags": self.tags_edit.toPlainText(),
            "log_dir": self.log_dir_edit.text().strip(),
            "tag_order": self.tag_order,
        }
        try:
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save settings: {e}", file=sys.stderr)

    # ---------------- polling control ----------------

    def choose_log_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Log Directory", self.log_dir_edit.text().strip() or "."
        )
        if path:
            self.log_dir = path
            self.log_dir_edit.setText(path)
            self.raw_csv_path = os.path.join(self.log_dir, "raw_data.csv")
            self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")
            ensure_csv(self.raw_csv_path, ["timestamp", "tag", "value", "status"])
            ensure_csv(
                self.hourly_csv_path,
                ["hour_start", "hour_end", "tag", "avg_value", "sample_count"],
            )
            self.log_message(f"Log directory set to: {path}")

    def start_polling(self):
        if self.poll_thread and self.poll_thread.isRunning():
            QMessageBox.warning(self, "Already running", "Polling is already active.")
            return

        ip = self.ip_edit.text().strip()
        if not ip:
            QMessageBox.warning(self, "Missing IP", "Please enter a PLC IP address.")
            return

        tags_text = self.tags_edit.toPlainText().strip()
        tags = [line.strip() for line in tags_text.splitlines() if line.strip()]
        if not tags:
            QMessageBox.warning(self, "No tags", "Please enter at least one tag to poll.")
            return

        if len(tags) > 200:
            QMessageBox.warning(
                self,
                "Many tags",
                f"You've configured {len(tags)} tags. The app can handle this, "
                "but consider keeping it to a few dozen for better responsiveness.",
            )

        self.tag_order = list(tags)

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
        self.update_dashboard()

        interval = self.interval_spin.value()
        chunk_size = self.chunk_spin.value()

        self.poll_thread = PollThread(
            ip=ip,
            tags=tags,
            interval_sec=interval,
            chunk_size=chunk_size,
            reconnect_delay_sec=10,
            parent=self,
        )
        self.poll_thread.data_ready.connect(self.handle_poll_data)
        self.poll_thread.error.connect(self.handle_poll_error)
        self.poll_thread.info.connect(self.log_message)
        self.poll_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.connection_status_label.setText("Status: Polling")
        self.connection_status_label.setStyleSheet("color: #a5d6a7;")
        self.log_message(
            f"Started polling every {interval} seconds for {len(tags)} tag(s)."
        )

    def stop_polling(self):
        if self.poll_thread:
            self.poll_thread.stop()
            self.poll_thread.wait(5000)
            self.poll_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.connection_status_label.setText("Status: Stopped")
        self.connection_status_label.setStyleSheet("color: #ef9a9a;")
        self.log_message("Stopped polling.")

    # ---------------- data handling ----------------

    def handle_poll_error(self, msg: str):
        """
        Handle errors emitted from PollThread.
        No modal message boxes here; just log and keep running.
        """
        self.log_message(msg)

    def handle_poll_data(self, ts_iso: str, data: Dict[str, Tuple[object, str]]):
        """
        Called from background thread with new readings.
        - Update current values
        - Append to raw CSV
        - Update hourly accumulators and roll over when hour changes
        """
        try:
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

            ensure_csv(self.raw_csv_path, ["timestamp", "tag", "value", "status"])

            try:
                with open(self.raw_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for tag, (val, status) in data.items():
                        self.current_values[tag] = (val, status)
                        writer.writerow([ts_iso, tag, val, status])

                        if isinstance(val, (int, float)) and str(status).lower() == "success":
                            acc = self.hour_accumulators.setdefault(
                                tag, {"sum": 0.0, "count": 0}
                            )
                            acc["sum"] += float(val)
                            acc["count"] += 1
            except Exception as e:
                self.log_message(f"Error writing raw CSV: {e}")

            self.last_update_label.setText(f"Last update: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            self.update_dashboard()
        except Exception as e:
            self.log_message(f"Error handling poll data: {e}")

    def finalize_previous_hour(self):
        """
        Compute averages for tags in the previous hour and write to hourly CSV.
        """
        if self.current_hour_start is None or not self.hour_accumulators:
            return

        hour_start = self.current_hour_start
        hour_end = hour_start + timedelta(hours=1)
        hour_start_iso = hour_start.isoformat(timespec="seconds")
        hour_end_iso = hour_end.isoformat(timespec="seconds")

        records: Dict[Tuple[str, str], Tuple[str, float, int]] = {}
        try:
            if os.path.exists(self.hourly_csv_path):
                with open(self.hourly_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = (row["hour_start"], row["tag"])
                        records[key] = (
                            row["hour_end"],
                            float(row["avg_value"]),
                            int(row.get("sample_count", "0") or 0),
                        )
        except Exception as e:
            self.log_message(f"Error reading hourly CSV for upsert: {e}")

        self.last_hour_avg.clear()
        for tag, acc in self.hour_accumulators.items():
            if acc["count"] <= 0:
                continue
            avg = acc["sum"] / acc["count"]
            self.last_hour_avg[tag] = avg
            key = (hour_start_iso, tag)
            records[key] = (hour_end_iso, avg, int(acc["count"]))

        try:
            ensure_csv(
                self.hourly_csv_path,
                ["hour_start", "hour_end", "tag", "avg_value", "sample_count"],
            )
            with open(self.hourly_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["hour_start", "hour_end", "tag", "avg_value", "sample_count"]
                )
                for (hs, tag), (he, avg, cnt) in sorted(records.items()):
                    writer.writerow([hs, he, tag, avg, cnt])
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
        if not os.path.exists(self.raw_csv_path):
            QMessageBox.warning(
                self, "No Raw Data", "Raw CSV does not exist; nothing to rebuild."
            )
            return

        self.log_message("Rebuilding hourly averages from raw data...")
        records: Dict[Tuple[str, str], Tuple[str, float, int]] = {}

        try:
            with open(self.raw_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                buckets: Dict[Tuple[datetime, str], Dict[str, float]] = {}
                for row in reader:
                    try:
                        ts = datetime.fromisoformat(row["timestamp"])
                    except Exception:
                        continue
                    tag = row["tag"]
                    status = str(row.get("status", "")).lower()
                    try:
                        val = float(row["value"])
                    except Exception:
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
                    hend_iso = (hstart + timedelta(hours=1)).isoformat(
                        timespec="seconds"
                    )
                    avg = acc["sum"] / acc["count"]
                    records[(hstart_iso, tag)] = (hend_iso, avg, int(acc["count"]))
        except Exception as e:
            self.log_message(f"Error rebuilding hourly from raw: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to rebuild hourly averages:\n{e}"
            )
            return

        try:
            ensure_csv(
                self.hourly_csv_path,
                ["hour_start", "hour_end", "tag", "avg_value", "sample_count"],
            )
            with open(self.hourly_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["hour_start", "hour_end", "tag", "avg_value", "sample_count"]
                )
                for (hs, tag), (he, avg, cnt) in sorted(records.items()):
                    writer.writerow([hs, he, tag, avg, cnt])
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
            Tag | Current Value | Last Hour Avg | Current Hour Avg (so far) | Status
        """
        if self.tag_order:
            tags = list(self.tag_order)
        else:
            tags = sorted(
                set(self.current_values.keys())
                | set(self.last_hour_avg.keys())
                | set(self.current_hour_preview.keys())
            )

        self.table.setRowCount(len(tags))
        self.rows_summary_label.setText(f"Rows: {len(tags)}")

        for row, tag in enumerate(tags):
            val, status = self.current_values.get(tag, ("", ""))

            last_avg = self.last_hour_avg.get(tag, "")
            cur_avg = self.current_hour_preview.get(tag, "")

            def item(text):
                it = QTableWidgetItem(str(text))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                return it

            self.table.setItem(row, 0, item(tag))
            self.table.setItem(row, 1, item(val))
            self.table.setItem(
                row,
                2,
                item(f"{last_avg:.4f}" if isinstance(last_avg, (int, float)) else ""),
            )
            self.table.setItem(
                row,
                3,
                item(f"{cur_avg:.4f}" if isinstance(cur_avg, (int, float)) else ""),
            )
            self.table.setItem(row, 4, item(status))

    # ---------------- logging & lifecycle ----------------

    def log_message(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_edit.appendPlainText(line)
        self.statusBar().showMessage(msg, 5000)

    def closeEvent(self, event):
        if self.poll_thread and self.poll_thread.isRunning():
            self.poll_thread.stop()
            self.poll_thread.wait(5000)
        self._save_settings()
        event.accept()


# ------------------- main entry -------------------


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
