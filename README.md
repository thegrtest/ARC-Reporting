# ARC-Reporting / CIP Tag Poller

This repository contains two coordinated tools for collecting and visualising PLC tag data:

- **`CIP.py`** – a PySide6 desktop application that polls ControlLogix/CompactLogix tags via `pylogix`, logs raw samples, and aggregates hourly averages while providing a live dark-themed dashboard.
- **`CIPMonitor.py`** – a Dash web server that renders gauges and tables from the log files produced by the poller so stakeholders can view current values, hourly averages, and exceedances in the browser.

## Project layout

- `CIP.py`: Main GUI poller and logger. Settings persist to `settings.json`; log outputs live in the `logs/` directory by default.
- `CIPMonitor.py`: Dash dashboard that reads `logs/raw_data_YYYY-MM-DD.csv`, `logs/hourly_averages.csv`, and supporting CSV/JSON files.
- `requirements.txt`: Python dependencies used by both applications.

## Prerequisites

- Python 3.9+ recommended.
- System packages for Qt (PySide6) may be required on Linux (e.g., `apt install libxkbcommon-x11-0 libglu1-mesa`).
- Install Python dependencies:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

Both applications will exit early with actionable messages if `pylogix`, `pandas`, `dash`, or `dash-daq` are missing.

## Running the poller GUI (`CIP.py`)

1. Ensure the PLC is reachable from your workstation and you know the IP address and tag names.
2. Start the application:

   ```bash
   python CIP.py
   ```

3. Populate the configuration fields:
   - **Machine/IP** – PLC identifier and IP address.
   - **Tags to Poll** – one tag per line. Add an optional alias after a pipe: `Program:MainRoutine.Flow_PV | Flow`.
   - **Interval/Chunk Size** – polling cadence (seconds) and batch size; chunk sizes above 10 auto-clamp to avoid PLC errors.【F:CIP.py†L9-L183】
   - **Machine state gating** – optional tag that controls whether polling pauses when the state is not `3 (Processing)`.【F:CIP.py†L29-L176】
   - **Heartbeat** – optional BOOL/DINT tag that is toggled each cycle so the PLC can watchdog the poller.【F:CIP.py†L42-L176】
   - **Log directory** – defaults to `logs/`; rotate/compress handled automatically.

4. Click **Start** to begin polling. The table displays current values, last full hour averages, and the current hour preview. Stop polling with **Stop**; settings are saved on exit.

### Output files

- **Raw samples**: `logs/raw_data_YYYY-MM-DD.csv` (one per day) with timestamp, tag, alias, value, and status.【F:CIP.py†L9-L14】
- **Hourly aggregates**: `logs/hourly_averages.csv`, updated in-place with upsert semantics (one row per tag per hour, plus an optional `avg_lb_hr` column for EPA conversions).【F:CIP.py†L11-L14】
- **System health**: `logs/system_health.json` tracks last poll times, disk space, error rate, and overall status for monitoring.【F:CIP.py†L130-L195】
- **Other diagnostics**: threshold configuration, environment events, and QA flags are preserved alongside the log files.

Use the **Rebuild Hourly Averages** button to regenerate `hourly_averages.csv` from all daily raw files if data is corrected or backfilled.【F:CIP.py†L1780-L1818】

## Running the web dashboard (`CIPMonitor.py`)

1. Ensure the poller has produced log files in `logs/` or copy existing logs there.
2. Start the Dash server:

   ```bash
   python CIPMonitor.py
   ```

3. Open `http://localhost:8050` (or replace `localhost` with the host IP for remote access). The dashboard shows gauges per tag for current value, last full hour average, and live current-hour average, colour-coded using `logs/thresholds.json`.【F:CIPMonitor.py†L5-L75】

The server reloads data every 5 seconds. It also exposes tabs for threshold management, configuration change history, exceedance review, and system health derived from `system_health.json`.【F:CIPMonitor.py†L55-L191】【F:CIPMonitor.py†L2001-L2040】

## Operational tips and recovery

- **Missing dependencies**: If the app exits complaining about `pylogix`, `pandas`, `dash`, or `dash-daq`, install them via `pip install -r requirements.txt` (or the specific package). The imports fail fast with clear messages.【F:CIP.py†L73-L78】【F:CIPMonitor.py†L34-L53】
- **PLC connection errors**: The poller automatically retries on connection failure and logs the error. Multi-read calls that hit "too many parameters" are auto-downgraded to single-tag reads.【F:CIP.py†L167-L199】【F:CIP.py†L148-L176】
- **Machine down handling**: Enable "Pause when machine down" to restrict polling to the machine-state tag until the state returns to Processing (`3`).【F:CIP.py†L36-L176】
- **Heartbeat watchdog**: Configure a heartbeat tag and enable heartbeat mode to let the PLC detect stalled polling; the value toggles every cycle.【F:CIP.py†L42-L176】
- **Log disk pressure**: The poller warns when the log folder exceeds a configured GB threshold and records disk free space in `system_health.json`. Prune or archive `logs/` if status degrades.【F:CIP.py†L130-L195】
- **Regenerating hourly averages**: If raw files were corrected or gaps filled, use the GUI button or rerun `CIP.py` and choose "Rebuild Hourly" to recompute aggregates from all `raw_data_*` files.【F:CIP.py†L1780-L1818】
- **Lockdown/autostart**: Enabling run lock disables configuration edits and automatically starts polling on launch. Disable it to adjust settings.【F:CIP.py†L1372-L1414】

## Development notes

- Main entry points: `python CIP.py` launches the PySide6 GUI; `python CIPMonitor.py` starts the Dash server on port 8050.【F:CIP.py†L1956-L1967】【F:CIPMonitor.py†L2037-L2040】
- Code style favors explicit logging and defensive error handling; keep try/except blocks tight around PLC or I/O calls.
- Tests are not provided; exercise the GUI and dashboard manually after changes.

## Common fixes checklist

- Verify `settings.json` contains the expected tags and IP address; delete it to start fresh if the UI loads with bad defaults.
- Clear out corrupted CSVs in `logs/` if parsing errors appear in the dashboard, then rebuild hourly averages.
- Confirm your user has permissions to create files under `logs/` and to open outbound connections to the PLC IP/port.
