# monitor.py
# Status checks and alerts from the Health Index. Owner: Person B.
# Interfaces follow section 16.7 of the project doc.

import numpy as np
import pandas as pd

from src.config import (ADVICE, COMPONENTS, HI_RESET_MARGIN, SMOOTH_DAYS,
                        STATUS_LEVELS, THRESHOLDS, X7_ADVICE, X7_BASELINE_DAYS,
                        X7_DROP, X7_RESET_DROP)
from src.health import health_table

ALERT_COLUMNS = ["DateTime", "component", "HI", "status", "message", "action"]


def check_status(hi):
    """Classify a Health Index.

    Args:
        hi (float): Health Index, 0 to 100.

    Returns:
        str: 'Normal' (hi >= 80), 'Warning' (50 <= hi < 80) or 'Critical'.

    Raises:
        ValueError: if hi is NaN or outside 0 to 100.
    """
    if np.isnan(hi) or not 0.0 <= hi <= 100.0:
        raise ValueError(f"Health Index must be between 0 and 100, got {hi}")
    if hi >= THRESHOLDS["Normal"]:
        return "Normal"
    if hi >= THRESHOLDS["Warning"]:
        return "Warning"
    return "Critical"


def _x7_status(drop):
    """Status from the x7 drop below baseline (fraction)."""
    if drop >= X7_DROP["Critical"]:
        return "Critical"
    if drop >= X7_DROP["Warning"]:
        return "Warning"
    return "Normal"


def scan(df):
    """Scan a Simulation DataFrame and raise an alert when a status gets worse.

    HI values are smoothed with a short rolling mean (SMOOTH_DAYS) so noise does
    not cause repeated alerts. An alert is made when a component first reaches
    Warning or Critical, and again only after it recovered clearly above the
    Normal threshold (HI_RESET_MARGIN), for example after repair. x7 is checked as a drop below its own baseline (first
    X7_BASELINE_DAYS days) and reported under the component 'turbine'.

    Args:
        df (pandas.DataFrame): Simulation DataFrame (section 16.5).

    Returns:
        pandas.DataFrame with DateTime, component, HI, status, message, action.
        Prints one line per alert.
    """
    table = health_table(df)
    hi_cols = [f"HI_{c}" for c in COMPONENTS]
    smooth = table[hi_cols].rolling(SMOOTH_DAYS, min_periods=1).mean()

    x7 = df["x7"].reset_index(drop=True)
    baseline = float(x7.iloc[:X7_BASELINE_DAYS].median())
    x7_drop = baseline - x7.rolling(SMOOTH_DAYS, min_periods=1).mean()

    rank = {s: i for i, s in enumerate(STATUS_LEVELS)}
    last = {c: 0 for c in COMPONENTS}
    last["x7"] = 0
    rows = []

    def emit(date, comp, hi, status, message, action):
        rows.append([date, comp, hi, status, message, action])
        print(f"[{date:%Y-%m-%d}] {status.upper()} - {message}")

    for i in range(len(table)):
        date = table.loc[i, "DateTime"]
        for comp in COMPONENTS:
            hi = float(smooth.loc[i, f"HI_{comp}"])
            status = check_status(hi)
            if hi >= THRESHOLDS["Normal"] + HI_RESET_MARGIN:
                last[comp] = 0            # recovered, alert can fire again
            elif rank[status] > last[comp]:
                last[comp] = rank[status]
                emit(date, comp, hi, status,
                     f"{comp} is {status} (HI = {hi:.0f})",
                     ADVICE[(comp, status)])
        status = _x7_status(float(x7_drop.iloc[i]))
        if x7_drop.iloc[i] < X7_RESET_DROP:
            last["x7"] = 0
        elif rank[status] > last["x7"]:
            last["x7"] = rank[status]
            hi_turb = float(smooth.loc[i, "HI_turbine"])
            emit(date, "turbine", hi_turb, status,
                 f"x7 is {x7_drop.iloc[i]:.3f} below baseline {baseline:.3f}",
                 X7_ADVICE[status])
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


if __name__ == "__main__":
    # Demo with mock data from C (run from the project root: python -m src.monitor)
    from src.mock_data import make_mock_df
    alerts = scan(make_mock_df(days=180, scenario="turbine_erosion"))
    print(alerts[["DateTime", "component", "status"]].to_string(index=False))
