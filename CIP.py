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
- Settings (IP, tags, interval, log dir) are remembered in settings.json.
"""

import sys
import os
import csv
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

from PySide6.QtCore import Qt, QThread, Signal, QMutex, QMutexLocker
from PySide6.QtGui import QPalette, QColor, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QPushButton, QTextEdit, QTableWidget,
    QTableWidgetItem, QMessageBox, QGroupBox, QFormLayout, QSpinBox,
    QFileDialog, QHeaderView, QPlainTextEdit, QStatusBar
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
    """

    data_ready = Signal(str, dict)
    error = Signal(str)
    info = Signal(str)

    def __init__(
        self,
        ip: str,
        tags: List[str],
        interval_sec: int = 10,
        chunk_size: int = 20,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.ip = ip
        self.tags = [t for t in tags if t.strip()]
        self.interval_sec = max(1, int(interval_sec))
        self.chunk_size = max(1, int(chunk_size))
        self._running = False
        self._mutex = QMutex()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    def _should_run(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def run(self):
        if not PYLOGIX_AVAILABLE:
            self.error.emit("pylogix is not installed. Run: pip install pylogix")
            return

        if not self.tags:
            self.error.emit("No tags specified to poll.")
            return

        with QMutexLocker(self._mutex):
            self._running = True

        try:
            with PLC() as comm:
                comm.IPAddress = self.ip
                self.info.emit(f"Connected to PLC at {self.ip}")

                while self._should_run():
                    start_t = time.time()
                    all_results: Dict[str, Tuple[object, str]] = {}

                    try:
                        for batch in chunked(self.tags, self.chunk_size):
                            res = comm.Read(*batch)
                            if not isinstance(res, list):
                                res = [res]
                            for r in res:
                                tag = getattr(r, "TagName", "(unknown)")
                                val = getattr(r, "Value", None)
                                st = getattr(r, "Status", "(unknown)")
                                all_results[tag] = (val, st)
                    except Exception as e:
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
            self.error.emit(f"PLC connection error: {e}")
        finally:
            self.info.emit("Polling thread stopped.")


# ------------------- main window -------------------


class MainWindow(QMainWindow):
    SETTINGS_FILE = "settings.json"

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CIP Tag Poller & Dashboard")
        self.resize(1200, 700)

        # --- state ---
        self.poll_thread: Optional[PollThread] = None
        self.current_values: Dict[str, Tuple[object, str]] = {}
        self.last_hour_avg: Dict[str, float] = {}
        self.current_hour_preview: Dict[str, float] = {}
        self.current_hour_start: Optional[datetime] = None
        self.hour_accumulators: Dict[str, Dict[str, float]] = {}

        self.log_dir = os.path.join("logs")
        self.raw_csv_path = os.path.join(self.log_dir, "raw_data.csv")
        self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")

        # UI
        self._build_ui()
        self._apply_dark_theme()
        self._load_settings()

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
        main_layout = QHBoxLayout(central)

        # Left side: config + controls + log
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, stretch=0)

        # Connection / tags group
        cfg_group = QGroupBox("Connection & Tags")
        cfg_layout = QFormLayout(cfg_group)

        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("192.168.1.20")
        cfg_layout.addRow(QLabel("PLC IP Address:"), self.ip_edit)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 3600)
        self.interval_spin.setValue(10)
        cfg_layout.addRow(QLabel("Poll Interval (seconds):"), self.interval_spin)

        tags_label = QLabel("Tags to Poll (one per line):")
        self.tags_edit = QTextEdit()
        self.tags_edit.setFixedHeight(120)
        self.tags_edit.setPlaceholderText("e.g.\nProgram:MainRoutine.MyTag1\nProgram:MainRoutine.MyTag2")
        cfg_layout.addRow(tags_label, self.tags_edit)

        # Log directory selector
        dir_layout = QHBoxLayout()
        self.log_dir_edit = QLineEdit(self.log_dir)
        browse_btn = QPushButton("Browse…")
        dir_layout.addWidget(self.log_dir_edit)
        dir_layout.addWidget(browse_btn)
        cfg_layout.addRow(QLabel("Log Directory:"), dir_layout)

        left_panel.addWidget(cfg_group)

        # Controls
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self.start_btn = QPushButton("Start Polling")
        self.stop_btn = QPushButton("Stop Polling")
        self.stop_btn.setEnabled(False)

        self.calc_btn = QPushButton("Compute Current Hour Avg (Display Only)")
        self.rebuild_btn = QPushButton("Rebuild Hourly CSV From Raw")

        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.calc_btn)
        ctrl_layout.addWidget(self.rebuild_btn)

        left_panel.addWidget(ctrl_group)

        # Event log
        log_group = QGroupBox("Event Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        left_panel.addWidget(log_group, stretch=1)

        # Status bar
        status = QStatusBar()
        self.setStatusBar(status)

        # Right side: dashboard table
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

        main_layout.addWidget(self.table, stretch=1)

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
        # Dark Fusion palette
        app = QApplication.instance()
        if app is None:
            return
        app.setStyle("Fusion")
        palette = QPalette()

        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)

        palette.setColor(QPalette.Highlight, QColor(64, 128, 255))
        palette.setColor(QPalette.HighlightedText, Qt.white)

        app.setPalette(palette)

        # Minimal stylesheet for groups & table
        app.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QTableWidget {
                gridline-color: #555;
            }
            QHeaderView::section {
                background-color: #333;
                color: white;
                padding: 4px;
                border: 1px solid #555;
            }
            QPushButton {
                padding: 6px 12px;
            }
            """
        )

    # ---------------- settings ----------------

    def _load_settings(self):
        if not os.path.exists(self.SETTINGS_FILE):
            return
        try:
            with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        self.ip_edit.setText(data.get("ip", ""))
        self.interval_spin.setValue(int(data.get("interval", 10)))
        self.tags_edit.setPlainText(data.get("tags", ""))
        self.log_dir = data.get("log_dir", self.log_dir)
        self.log_dir_edit.setText(self.log_dir)
        self.raw_csv_path = os.path.join(self.log_dir, "raw_data.csv")
        self.hourly_csv_path = os.path.join(self.log_dir, "hourly_averages.csv")

    def _save_settings(self):
        data = {
            "ip": self.ip_edit.text().strip(),
            "interval": self.interval_spin.value(),
            "tags": self.tags_edit.toPlainText(),
            "log_dir": self.log_dir_edit.text().strip(),
        }
        try:
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

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

        # reset per-run state
        self.current_values.clear()
        self.current_hour_preview.clear()
        self.current_hour_start = None
        self.hour_accumulators.clear()
        # NOTE: last_hour_avg is not cleared so we still show last known hour
        self.update_dashboard()

        interval = self.interval_spin.value()

        self.poll_thread = PollThread(
            ip=ip,
            tags=tags,
            interval_sec=interval,
            chunk_size=20,
            parent=self,
        )
        self.poll_thread.data_ready.connect(self.handle_poll_data)
        self.poll_thread.error.connect(self.handle_poll_error)
        self.poll_thread.info.connect(self.log_message)
        self.poll_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_message(f"Started polling every {interval} seconds.")

    def stop_polling(self):
        if self.poll_thread:
            self.poll_thread.stop()
            self.poll_thread.wait(5000)
            self.poll_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_message("Stopped polling.")

    # ---------------- data handling ----------------

    def handle_poll_error(self, msg: str):
        self.log_message(msg)
        QMessageBox.warning(self, "Polling Error", msg)
        self.stop_polling()

    def handle_poll_data(self, ts_iso: str, data: Dict[str, Tuple[object, str]]):
        """
        Called from background thread with new readings.
        - Update current values
        - Append to raw CSV
        - Update hourly accumulators and roll over when hour changes
        """
        try:
            ts = datetime.fromisoformat(ts_iso)
        except Exception:
            ts = datetime.now()

        hour_start = hour_bucket(ts)

        if self.current_hour_start is None:
            self.current_hour_start = hour_start
        elif hour_start != self.current_hour_start:
            # Hour changed: finalize previous hour and start a new bucket
            self.finalize_previous_hour()
            self.current_hour_start = hour_start
            self.hour_accumulators.clear()
            self.current_hour_preview.clear()

        ensure_csv(self.raw_csv_path, ["timestamp", "tag", "value", "status"])

        # Update current values and accumulators, write raw CSV
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

        self.update_dashboard()

    def finalize_previous_hour(self):
        """
        Compute averages for tags in the previous hour and write to hourly CSV
        with UPSERT semantics (one row per (hour_start, tag)).
        Also cache as last_hour_avg for UI.
        """
        if self.current_hour_start is None or not self.hour_accumulators:
            return

        hour_start = self.current_hour_start
        hour_end = hour_start + timedelta(hours=1)
        hour_start_iso = hour_start.isoformat(timespec="seconds")
        hour_end_iso = hour_end.isoformat(timespec="seconds")

        # load existing hourly data into dict keyed by (hour_start_iso, tag)
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

        # update/insert current hour
        self.last_hour_avg.clear()
        for tag, acc in self.hour_accumulators.items():
            if acc["count"] <= 0:
                continue
            avg = acc["sum"] / acc["count"]
            self.last_hour_avg[tag] = avg
            key = (hour_start_iso, tag)
            records[key] = (hour_end_iso, avg, int(acc["count"]))

        # rewrite CSV
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
        """
        Compute current hour averages based on data accumulated so far.
        Does NOT write to hourly CSV; only updates UI.
        """
        self.current_hour_preview.clear()
        for tag, acc in self.hour_accumulators.items():
            if acc["count"] <= 0:
                continue
            self.current_hour_preview[tag] = acc["sum"] / acc["count"]
        self.update_dashboard()
        self.log_message("Computed current hour averages (display only).")

    def rebuild_hourly_from_raw(self):
        """
        Recompute all hourly averages from raw_data.csv and rewrite hourly_averages.csv.
        Uses UPSERT semantics automatically by construction.
        """
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

        # rewrite hourly CSV
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
        tags = sorted(
            set(self.current_values.keys())
            | set(self.last_hour_avg.keys())
            | set(self.current_hour_preview.keys())
        )
        self.table.setRowCount(len(tags))

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
        # ensure thread stops
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
