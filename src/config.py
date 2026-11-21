# config.py
# Design values, failure points, thresholds and advice text for the Health Index.
# Owner: Person B. Shapes follow sections 16.4 - 16.7 of the project doc.
# Every number marked ASSUMPTION is a tunable demo value (step B5), not plant data.

# Measured variable -> value at which the component is healthy (HI = 100)
DESIGN = {
    "eta_turb": 0.88,   # ASSUMPTION, typical range 0.85 to 0.90
    "eta_p1": 0.85,     # ASSUMPTION, typical range 0.75 to 0.85
    "eta_p2": 0.85,     # ASSUMPTION
    "T5": 600.0,        # C, from the diagram
    "P7": 0.010,        # MPa, from the diagram
}

# Measured variable -> value at which the component counts as failed (HI = 0)
FAIL = {
    "eta_turb": 0.70,   # ASSUMPTION
    "eta_p1": 0.60,     # ASSUMPTION
    "eta_p2": 0.60,     # ASSUMPTION
    "T5": 540.0,        # C, ASSUMPTION (same as the boiler-fouling floor)
    "P7": 0.020,        # MPa, ASSUMPTION (higher is worse)
}

# Exact component names (16.6) and the measured column each one uses
COMPONENT_VARIABLE = {
    "turbine": "eta_turb",
    "pump1": "eta_p1",
    "pump2": "eta_p2",
    "boiler": "T5",
    "condenser": "P7",
}
COMPONENTS = tuple(COMPONENT_VARIABLE)

# Exact status labels (16.6), from best to worst
STATUS_LEVELS = ("Normal", "Warning", "Critical")

# Health Index bands: Normal >= 80, Warning 50 to 80, Critical < 50
THRESHOLDS = {"Normal": 80, "Warning": 50}

# Remaining life is the time until HI falls to this value
RUL_THRESHOLD = THRESHOLDS["Warning"]

# Smoothing window (days) used by monitor.scan so sensor noise does not
# make the status flicker. ASSUMPTION.
SMOOTH_DAYS = 5

# Hysteresis: after an alert, a component must recover this far above the Normal
# threshold (HI points) before it can raise a new alert. ASSUMPTION.
HI_RESET_MARGIN = 5
X7_RESET_DROP = 0.01   # x7 drop (fraction) below which the x7 alert is re-armed

# x7 rule (16.5): alert on a DROP below the baseline, not on the absolute value,
# because the design x7 is already below the 0.88 blade guideline.
X7_BASELINE_DAYS = 7                           # first days of the run = baseline
X7_DROP = {"Warning": 0.02, "Critical": 0.05}  # fraction below baseline

ADVICE = {
    ("turbine", "Normal"): "No action. Keep routine monitoring.",
    ("turbine", "Warning"): "Plan a blade inspection at the next outage; check for erosion and deposits.",
    ("turbine", "Critical"): "Schedule turbine overhaul or blade repair now; efficiency loss is large.",
    ("pump1", "Normal"): "No action. Keep routine monitoring.",
    ("pump1", "Warning"): "Check Pump I impeller wear, seals and suction conditions at next outage.",
    ("pump1", "Critical"): "Repair or replace Pump I now; run the standby pump if available.",
    ("pump2", "Normal"): "No action. Keep routine monitoring.",
    ("pump2", "Warning"): "Check Pump II wear and cavitation; review feedwater supply.",
    ("pump2", "Critical"): "Repair or replace Pump II now; high-pressure duty makes losses costly.",
    ("boiler", "Normal"): "No action. Keep routine monitoring.",
    ("boiler", "Warning"): "Plan soot blowing or cleaning of boiler heat-transfer surfaces.",
    ("boiler", "Critical"): "Shut down for boiler tube cleaning; steam temperature is far below design.",
    ("condenser", "Normal"): "No action. Keep routine monitoring.",
    ("condenser", "Warning"): "Check cooling-water flow and clean condenser tubes; look for air leakage.",
    ("condenser", "Critical"): "Clean the condenser now; back pressure is near the failure limit.",
}

X7_ADVICE = {
    "Warning": "Exhaust steam is getting wetter; inspect turbine and check boiler and condenser conditions.",
    "Critical": "Exhaust moisture is high; blade erosion risk. Find the cause and repair soon.",
}
