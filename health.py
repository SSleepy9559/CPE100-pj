# health.py
# Health Index (0-100), per-component health, health table and remaining life.
# Owner: Person B. Interfaces follow section 16.7 of the project doc.

import numpy as np
import pandas as pd

from src.config import COMPONENT_VARIABLE, COMPONENTS, DESIGN, FAIL, RUL_THRESHOLD


def health_index(measured, design, fail):
    """Convert a measured value to a Health Index from 0 (failed) to 100 (design).

    Works when higher is better (efficiency: design > fail) and when higher is
    worse (condenser pressure: design < fail), because the sign cancels.

    Args:
        measured (float, array or Series): measured value, same unit as design.
        design (float): value of a healthy component (HI = 100).
        fail (float): value of a failed component (HI = 0).

    Returns:
        float for a single value, otherwise an array or Series of floats, 0 to 100.

    Raises:
        ValueError: if design equals fail.
    """
    if design == fail:
        raise ValueError("design and fail must be different numbers")
    if isinstance(measured, (list, tuple)):
        measured = np.asarray(measured, dtype=float)
    ratio = (measured - fail) / (design - fail)
    if isinstance(ratio, pd.Series):
        return 100.0 * ratio.clip(0.0, 1.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    return float(100.0 * ratio) if np.ndim(ratio) == 0 else 100.0 * ratio


def _check_columns(columns):
    """Raise ValueError if a measured column needed for health is missing."""
    missing = [v for v in COMPONENT_VARIABLE.values() if v not in columns]
    if missing:
        raise ValueError(f"missing columns for health: {missing}")


def component_health(row):
    """Health Index of every component for one row of the Simulation DataFrame.

    Args:
        row (dict or pandas.Series): one row with T5 (C), P7 (MPa),
            eta_turb, eta_p1, eta_p2 (fractions).

    Returns:
        dict: keys turbine, pump1, pump2, boiler, condenser; values 0 to 100.
    """
    _check_columns(row.keys())
    return {
        comp: health_index(float(row[var]), DESIGN[var], FAIL[var])
        for comp, var in COMPONENT_VARIABLE.items()
    }


def health_table(df):
    """Health Index of every component for every row (day).

    Args:
        df (pandas.DataFrame): Simulation DataFrame (section 16.5).

    Returns:
        pandas.DataFrame with DateTime, day, HI_turbine, HI_pump1, HI_pump2,
        HI_boiler, HI_condenser, HI_overall. HI_overall is the lowest component
        HI (weakest link). All HI_ columns are floats from 0 to 100.
    """
    _check_columns(df.columns)
    out = df[["DateTime", "day"]].copy()
    for comp, var in COMPONENT_VARIABLE.items():
        out[f"HI_{comp}"] = health_index(df[var], DESIGN[var], FAIL[var])
    out["HI_overall"] = out[[f"HI_{c}" for c in COMPONENTS]].min(axis=1)
    return out.reset_index(drop=True)


def remaining_life(days, hi_series, threshold=RUL_THRESHOLD):
    """Estimate days until the Health Index falls to a threshold.

    Fits HI(t) with a straight line (np.polyfit) and extrapolates.

    Args:
        days (array-like): day numbers (days).
        hi_series (array-like): Health Index values, same length as days.
        threshold (float): HI level that counts as end of life.

    Returns:
        float: days from the last given day until the threshold (0.0 if already
        at or below it), or None if HI is not declining or data is too short.

    Raises:
        ValueError: if days and hi_series have different lengths.
    """
    days = np.asarray(days, dtype=float)
    hi = np.asarray(hi_series, dtype=float)
    if days.shape != hi.shape:
        raise ValueError("days and hi_series must have the same length")
    ok = ~(np.isnan(days) | np.isnan(hi))
    days, hi = days[ok], hi[ok]
    if len(days) < 2:
        return None
    slope, intercept = np.polyfit(days, hi, 1)
    if slope >= 0:
        return None
    if slope * days[-1] + intercept <= threshold:
        return 0.0
    return float((threshold - intercept) / slope - days[-1])


if __name__ == "__main__":
    # Demo with mock data from C (run from the project root: python -m src.health)
    from src.mock_data import make_mock_df
    demo = make_mock_df(days=180, scenario="turbine_erosion")
    table = health_table(demo)
    print(table.iloc[[0, 60, 120, 179]].round(1).to_string(index=False))
    print("remaining life (days):", remaining_life(table["day"], table["HI_turbine"]))
