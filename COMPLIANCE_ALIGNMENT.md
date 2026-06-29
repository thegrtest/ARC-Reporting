# SN-31B Compliance Alignment Summary
**Permit:** ADEQ Title V, Permit No. 1272-AOP-R1, AFIN 43-00024  
**Source:** SN-31B — Rotary Kiln with Air Pollution Control (APC)  
**Regulatory basis:** 40 CFR Part 63 Subpart EEE (HWC MACT, new source); ADEQ Rule 19  
**Date prepared:** 2026-06-29  

---

## 1. Confirmed Emission Limits (SN-31B)

### Mass-rate limits (SC#130 — enforceable specific conditions)
| Pollutant | Limit (lb/hr) | Limit (tpy) | Averaging Period | App Tag |
|-----------|-------------|-------------|-----------------|---------|
| PM10      | 0.1         | 0.3         | Performance test | — |
| SO2       | 0.4         | 1.6         | Performance test | — |
| VOC       | 1.2         | 5.1         | — | — |
| **CO**    | **1.6**     | **6.7**     | Hourly rolling avg (CEMS) | `EPA19:CO_LBHR` |
| **NOx**   | **10.2**    | **38.3**    | **12-hour rolling average** | `EPA19:NOx_LBHR` |
| Lead      | 1.31E-04    | 5.74E-04    | Performance test | — |

> Note: Emission summary table (permit page 15) shows NOx at 10.6 lb/hr.  
> SC#130 table is the enforceable limit — **10.2 lb/hr** governs.

### EEE new-source concentration limits (SC#216, 40 CFR 63.1219(b))
All corrected to **7% O2**, dry basis:
| Pollutant | EEE Limit | Averaging Period |
|-----------|-----------|-----------------|
| **CO**    | **100 ppmv** | Hourly rolling average |
| HC (propane) | 10 ppmv | Hourly rolling average |
| PM        | 0.0016 gr/dscf (≈3.7 mg/dscm) | Performance test |
| D/F       | 0.20 ng TEQ/dscm | Performance test |
| Mercury   | 8.1 μg/dscm | — |
| Cd + Pb   | 10 μg/dscm (combined) | — |
| As + Be + Cr | 23 μg/dscm (combined) | — |
| HCl + Cl₂ | 21 ppmv (as Cl⁻ equivalent) | — |
| DRE       | 99.99% | Performance test |

### Operating limits
| Requirement | Limit | SC# | Tag |
|-------------|-------|-----|-----|
| Throughput  | ≤ 1,200 lb/hr | SC#136 | `RotaryKilnData[10]` |
| Annual burn hours | ≤ 7,500 hr/yr | SC#137 | Derived from machine state |
| **Afterburner temp** | **≥ 1,700°F** | SC#139 | `RotaryKilnData[4]` |
| VOC         | ≤ 10 ppmv | SC#130 footnote | — |
| Opacity     | ≤ 5% | SC#132 | — |

---

## 2. Monitoring Requirements

### Required CEMS (per permit)
| Monitor | Requirement | SC# | App tag |
|---------|------------|-----|---------|
| CO CEMS | Continuous, per Dept. CEMS Conditions | SC#138 | `RotaryKilnData[1]` |
| O2 CEMS | Required by EEE to correct CO to 7% O2 | SC#183 | `RotaryKilnData[0]` |
| NOx CEMS | Continuous, 12-hr rolling avg in lb/hr | SC#140 | `RotaryKilnData[2]` |
| PM CEMS | Required by EEE; deferred until EPA promulgates specs (SC#184) | SC#184 | — |
| Afterburner temp CPMS | Continuous parameter monitor | SC#139 | `RotaryKilnData[4]` |

### NOx rolling average — permit vs. app
**SC#140 definition:** "arithmetic mean of the 720 one-minute average values [of concentration]… convert the concentrations recorded by the CEMs to lb/hr."  
**App implementation:** minute-level concentration accumulators → 12-hr **mean concentration** → convert to lb/hr (EPA Method 19) using the 12-hr mean flow, applying the CEMS data-availability rule (≥75% data capture required per hour).

> **Alignment note (resolved 2026-06-29):** The permit-literal method is "average the one-minute concentrations over the window, then convert to lb/hr." The collector now follows this exactly. Previously the persisted rolling CSV averaged the *per-hour lb/hr* values while the rebuild and live-snapshot paths derived lb/hr from the mean concentration — three code paths that could disagree when flue-gas flow varied hour to hour. All three were unified to the permit-literal **mean-concentration → convert** method, so the value persisted to disk, the value written back to the PLC, and any rebuilt value are now identical for a given 12-hr window. (CIP.py: `_update_rolling_12hr_from_records`, `_rebuild_rolling_12hr_from_records`, `_compute_live_rolling_12hr`.)
>
> **Still confirm with the CEMS integrator** whether the official compliance record of record is this DAS/PLC value or the CEMS vendor's own system, and whether the conversion to lb/hr should use the 12-hr mean flow (current app behavior) or one-minute flow paired with one-minute concentration (see CEMS Q6/Q7).

---

## 3. App Configuration Alignment

### Changes applied (2026-06-29)
The following limits were added to `logs/thresholds.json`:

| Tag | Added limit | Basis |
|-----|-------------|-------|
| `EPA19:NOx_LBHR` | `high_limit: 10.2` | SC#130, 12-hr rolling avg |
| `EPA19:CO_LBHR`  | `high_limit: 1.6`  | SC#130, hourly avg |
| `RotaryKilnData[4]` (Afterburner Temp) | `low_limit: 1700` | SC#139 (°F) |
| `RotaryKilnData[10]` (Hourly Weight Total) | `high_limit: 1200` | SC#136 (lb/hr) |

These limits activate:
- Yellow/red gauge zones on the dashboard
- Exceedance detection in compliance % calculations (24-hr, 30-day)
- Flagging in Compliance and Incident report types

### Confirmed alignments
- **12-hr rolling average period**: Permit (SC#140) requires 12-hr. App uses 12-hr. ✓
- **EPA Method 19 molar volume**: App uses 385.3 dscf/lb-mol (68°F / 29.92 inHg). ✓
- **NOx molecular weight**: 46.0 g/mol (as NO₂). ✓
- **CO molecular weight**: 28.01 g/mol. ✓
- **Data-capture rule**: App requires ≥75% data capture per hour (≥45 of 60 minutes) before including an hour in rolling averages. Sparse hours are excluded entirely — consistent with CEMS data-availability practices. ✓
- **Machine state**: Operating state defined as `RotaryKiln_OperationState` value in [2.5, 3.5] range (state 3 = Processing). Consistent with SC#137 burn hour tracking. ✓
- **Raw data compression**: Files compressed to .csv.gz after 7 days; all history functions read both formats. ✓

---

## 4. CEMS Configuration Questions

These must be answered before the compliance reporting can be fully validated.  
**Flag to CEMS integrator / DAS vendor:**

### Q1 — CO CEMS units
**`RotaryKilnData[1]` (CEMS CO) — what units does the PLC tag output?**  
- ppmv (dry basis)? ppmv corrected to 7% O2? % volume? mg/m³?  
- The EEE standard (CO ≤ 100 ppmv) requires the value corrected to 7% O2, dry basis.  
- The lb/hr calculation in the app uses: `ppm × flow_dscfm × 60 × MW / (1e6 × 385.3)`  
- If the CO CEMS is already outputting 7%-O2-corrected ppmv, the lb/hr computed by the app already reflects corrected mass. If not, a correction factor is needed.

### Q2 — NOx CEMS units and species
**`RotaryKilnData[2]` (CEMS NOx) — what units and species?**  
- ppmv as NO? ppmv as NO₂ (equivalent)? ppmv total NOx?  
- App assumes ppmv and uses MW = 46.0 g/mol (as NO₂) per EPA Method 19.  
- If the CEMS outputs as NO (MW 30.0), the lb/hr will be underestimated by ~35%.  
- **Confirm: does the CEMS output NOx as NO₂ equivalent?**

### Q3 — O2 reference for lb/hr calculation
**Is O2 dilution correction applied in the EPA Method 19 lb/hr formula?**  
- The app formula: `ppm × flow_dscfm × 60 × MW / (1e6 × 385.3)` does NOT apply O2 correction.  
- EPA Method 19 for lb/hr does NOT require O2 correction — O2 correction is only needed for ppmv compliance comparisons at a reference O2 %.  
- However, if flow is measured as dry standard flow (dscfm) at actual stack conditions, and CEMS output is dry ppmv at actual O2, the lb/hr is correct without O2 correction.  
- **Confirm: `RotaryKilnData[7]` (System Flue Gas Flow) — is this in dscfm (dry, at standard temperature/pressure)?**

### Q4 — Flow tag units and measurement method
**`RotaryKilnData[7]` (System Flue Gas Flow) — what units and method?**  
- Must be in dscfm for the EPA Method 19 lb/hr formula to be correct.  
- If in acfm (actual cubic feet per minute, not dry/standard), a temperature and moisture correction is needed: dscfm = acfm × (520/T_°R) × (P/29.92) × (1 - %moisture/100).  
- If a pitot/S-probe flow, confirm it's corrected to standard conditions (68°F, 29.92 inHg).

### Q5 — CO CEMS span value
**What is the CO CEMS span value?**  
- Per EEE Performance Specification PS-4B: span must be 3,000 ppmv.  
- If a one-minute CO reading hits or exceeds the span (3,000 ppmv), it must be recorded as 10,000 ppmv for the hourly rolling average calculation (per SC#185).  
- **Confirm: does the DAS/PLC implement this clamp?**

### Q6 — PM CEMS status
**What does `RotaryKilnData[6]` (Baghouse Leak Detection) output?**  
- Is this a triboelectric/opacity alarm (binary: 0/1) or a continuous numeric response?  
- The PM CEMS correlation (alarm setpoint = 50% of PM standard or 125% of highest correlation PM) has not yet been established (PM CEMS specs pending, per SC#184).  
- Until the PM CEMS alarm setpoint is formally established via a comprehensive performance test, the Baghouse Leak Detection alarm is an operations indicator only — not a compliance limit threshold.

### Q7 — NOx averaging: ppm vs. lb/hr
**Does the DAS calculate 12-hr rolling average in ppmv first, then convert to lb/hr?**  
- Permit SC#140: record 12-hr average concentrations (ppmv), then convert to lb/hr.  
- App: computes lb/hr each minute, then averages 12 hours of lb/hr values.  
- If flow varies within the hour, these methods give different results.  
- **Confirm with CEMS integrator which method the permit DAS uses for official compliance records.**

### Q8 — EEE CO 7% O2 correction in real-time
**Is the CO CEMS performing continuous 7% O2 correction in real-time?**  
- EEE requires CO measured and reported at 7% O2 reference.  
- O2 correction: C_corrected = C_measured × (21 - 7) / (21 - %O2_actual) = C_measured × 14 / (21 - O2_actual)  
- If the PLC tag `RotaryKilnData[1]` is already 7%-O2-corrected ppmv, the 100 ppmv EEE limit applies directly.  
- If it's uncorrected ppmv, the corrected concentration could be higher or lower than the measured value depending on actual O2 %, and the limit check needs to apply the correction.  
- **The app currently treats the raw CO tag value as ppmv without O2 correction.**

---

## 5. Annual Burn Hours Tracking

SC#137 limits SN-31B to **7,500 burn hours per year**.  

The app tracks processing time based on `RotaryKiln_OperationState` = 3 (Processing). The CLI `python CIPMonitor.py metrics all_time` command reports total processing minutes. Converting:

- Current all-time processing: 40,816 minutes = **680.3 hours**  
- Limit: 7,500 hours/year  

The app does not yet generate an annual burn-hours report. This should be added or tracked separately. The operational summary report already shows "Hours Processed" for any date range — running the `prev_month` or `month` range monthly will cover this.

**Action needed:** Confirm the date the kiln went into operation to establish the 12-month rolling window for the 7,500 hr/yr limit.

---

## 6. Summary: What Is Confirmed vs. What Needs Follow-Up

### Confirmed / implemented
| Item | Status |
|------|--------|
| 12-hr rolling NOx average | ✓ Implemented, matches permit SC#140 |
| NOx lb/hr limit 10.2 in thresholds.json | ✓ Added 2026-06-29 |
| CO lb/hr limit 1.6 in thresholds.json | ✓ Added 2026-06-29 |
| Afterburner temp minimum 1700°F in thresholds.json | ✓ Added 2026-06-29 |
| Throughput limit 1200 lb/hr in thresholds.json | ✓ Added 2026-06-29 |
| EPA Method 19 molar volume 385.3 dscf/lb-mol | ✓ Correct in both CIP.py and CIPMonitor.py |
| NOx MW as NO₂ (46.0 g/mol) | ✓ In code — confirm CEMS output is also as NO₂ |
| Data-capture rule (≥75%/hr) for rolling averages | ✓ Implemented |
| Warm-start hour accumulators on restart (incl. over-limit data) | ✓ Implemented + hardened 2026-06-29 |
| Atomic CSV writes (no truncation on crash) | ✓ Implemented |
| Rolling 12-hr lb/hr unified to permit-literal mean-conc→convert (all 3 paths) | ✓ Implemented 2026-06-29 |
| Restart hour-boundary gap recovery | ✓ Implemented 2026-06-29 |
| CO span substitution (3,000→10,000 ppmv, PS-4B/SC#185) scoped to CO only | ✓ Implemented 2026-06-29 |
| 12-hr average method (mean concentration → convert to lb/hr) | ✓ Implemented per SC#140; confirm record-of-record with integrator |

### Needs CEMS integrator input
| Item | Blocker |
|------|---------|
| CO ppmv limit (100 ppmv @ 7% O2) in thresholds | Need to confirm CO tag units and whether 7% O2 correction is pre-applied |
| NOx ppmv limit (if applicable) | Need to confirm NOx tag units and species (NO vs NO₂) |
| Flow tag units (dscfm vs. acfm) | Determines if EPA Method 19 lb/hr is correct without additional correction |
| CO CEMS span clamp value | Cap is implemented (CO-only, 3,000→10,000); confirm the CO CEMS span is actually 3,000 ppmv per PS-4B |
| PM CEMS alarm setpoint | Deferred until EPA promulgates PS; baghouse leak detection is ops indicator only |
| 12-hr average record-of-record | App now matches SC#140 (mean conc → convert); confirm whether DAS/PLC value or CEMS-vendor system is the official record, and mean-flow vs one-minute-flow conversion |

---

## 7. Collector Hardening (CIP.py, 2026-06-29)

Following an independent audit of the data collector, four correctness fixes were
applied and re-verified by adversarial review (all confirmed correct, no regressions):

1. **Warm-start no longer drops over-limit data.** On a mid-hour restart, the
   in-progress hour is reconstructed from raw using the same gate as live
   collection (`ACCUMULATABLE_QA_FLAGS` = OK + OUT_OF_RANGE + MANUAL_CORRECTION),
   so the highest-emission readings are retained rather than discarded. The
   per-minute average also passes through the same CO span cap as the live path.

2. **Rolling 12-hr lb/hr unified to the permit-literal method.** All three
   computation paths (persisted CSV, PLC write-back, and rebuild) now derive
   lb/hr from the 12-hr **mean concentration × mean flow** per SC#140, instead of
   one path averaging per-hour lb/hr. The persisted value, the value sent to the
   PLC, and any rebuilt value are now identical for the same window.

3. **Restart hour-boundary gap recovery.** If the poller stops mid-hour and
   restarts in a later hour, the previously in-progress (now completed) hour is
   recovered from raw and finalized into `hourly_averages.csv` on startup, so a
   restart cannot leave a hole in the 12-hr window. The recovery is additive-only
   (never modifies existing hours), bounded to a 2-day lookback, and fully
   fault-isolated. Longer outages should be repaired with "Rebuild hourly from raw."

4. **CO span substitution scoped to CO and documented.** The 3,000→10,000 ppmv
   one-minute substitution (EPA PS-4B, permit SC#185) now applies to the CO CEMS
   only — it has no regulatory basis for NOx, so NOx readings are carried through
   honestly.

These changes affect only how the collector reconstructs and averages data on
restart and how the rolling value is computed; the report-generation side
(CIPMonitor.py) was independently re-verified to still reconcile to full
precision against the raw CSVs after the changes.
