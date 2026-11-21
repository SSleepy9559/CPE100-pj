# validate_vs_coolprop.py
# Compares our steam_props.py (CSV tables + interpolation) with CoolProp (IAPWS).
# Owner: Person D. CoolProp is allowed only in scripts/ (16.2).
# Run from the project root:  python scripts/validate_vs_coolprop.py
# Done when: max percent error is under about 0.5 percent (15.6).

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable when this file is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.steam_props import props_PT, sat_props  # noqa: E402

try:
    from CoolProp.CoolProp import PropsSI
except ImportError:  # CoolProp is only needed for this script
    PropsSI = None

# ---------------------------------------------------------------- settings
N_POINTS = 50            # number of random (P, T) points (15.6)
SEED = 1                 # shared random seed (16.5)
P_RANGE_MPA = (0.01, 15.0)   # ASSUMPTION: must match the range of data/sh_table.csv
T_MAX_C = 600.0              # ASSUMPTION: must match the range of data/sh_table.csv
SUPERHEAT_MARGIN_C = 10.0    # stay away from the saturation line, where tables are rough
SAT_PRESSURES_MPA = (0.01, 0.05, 0.1, 0.5, 1.2, 3.0, 5.0, 10.0, 15.0)  # saturation check
TOLERANCE_PERCENT = 0.5      # pass limit (15.6)
FLUID = "Water"


def _require_coolprop():
    """Raise a clear error if CoolProp is not installed."""
    if PropsSI is None:
        raise ImportError("CoolProp is not installed. Run: pip install CoolProp")


def percent_error(ours, reference):
    """Absolute percent error of a value against a reference (same units)."""
    return abs(ours - reference) / abs(reference) * 100.0


def validate_superheated(n=N_POINTS, seed=SEED):
    """Compare props_PT with CoolProp at random superheated (P, T) points.

    Args:
        n (int): number of random points.
        seed (int): random seed, so the same points are used every run.

    Returns:
        pandas.DataFrame with one row per point: P (MPa), T (C), and the percent
        error of h, s and v.
    """
    _require_coolprop()
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        p_mpa = rng.uniform(*P_RANGE_MPA)
        t_sat = PropsSI("T", "P", p_mpa * 1e6, "Q", 1, FLUID) - 273.15
        t_low = t_sat + SUPERHEAT_MARGIN_C
        if t_low >= T_MAX_C:
            continue
        t_c = rng.uniform(t_low, T_MAX_C)
        ours = props_PT(p_mpa, t_c)
        args = ("P", p_mpa * 1e6, "T", t_c + 273.15, FLUID)
        h_ref = PropsSI("H", *args) / 1000.0      # kJ/kg
        s_ref = PropsSI("S", *args) / 1000.0      # kJ/(kg.K)
        v_ref = 1.0 / PropsSI("D", *args)         # m3/kg
        rows.append({
            "P_MPa": p_mpa, "T_C": t_c,
            "err_h_%": percent_error(ours["h"], h_ref),
            "err_s_%": percent_error(ours["s"], s_ref),
            "err_v_%": percent_error(ours["v"], v_ref),
        })
    return pd.DataFrame(rows)


def validate_saturation(pressures=SAT_PRESSURES_MPA):
    """Compare sat_props with CoolProp at fixed pressures.

    Args:
        pressures (iterable of float): pressures (MPa).

    Returns:
        pandas.DataFrame with P (MPa) and the percent error of Tsat, hf, hg, sf, sg.
    """
    _require_coolprop()
    rows = []
    for p_mpa in pressures:
        ours = sat_props(p_mpa)
        p_pa = p_mpa * 1e6
        ref = {
            "Tsat": PropsSI("T", "P", p_pa, "Q", 0, FLUID) - 273.15,
            "hf": PropsSI("H", "P", p_pa, "Q", 0, FLUID) / 1000.0,
            "hg": PropsSI("H", "P", p_pa, "Q", 1, FLUID) / 1000.0,
            "sf": PropsSI("S", "P", p_pa, "Q", 0, FLUID) / 1000.0,
            "sg": PropsSI("S", "P", p_pa, "Q", 1, FLUID) / 1000.0,
        }
        row = {"P_MPa": p_mpa}
        for key, value in ref.items():
            row[f"err_{key}_%"] = percent_error(ours[key], value)
        rows.append(row)
    return pd.DataFrame(rows)


def error_table(df):
    """Summary of the error columns: mean and max percent error of each property.

    Args:
        df (pandas.DataFrame): output of validate_superheated or validate_saturation.

    Returns:
        pandas.DataFrame with columns property, mean_%, max_%.
    """
    cols = [c for c in df.columns if c.startswith("err_")]
    return pd.DataFrame({
        "property": [c[4:-2] for c in cols],
        "mean_%": [df[c].mean() for c in cols],
        "max_%": [df[c].max() for c in cols],
    })


def passed(df, tolerance=TOLERANCE_PERCENT):
    """True if every error in df is below the tolerance (percent)."""
    cols = [c for c in df.columns if c.startswith("err_")]
    return bool((df[cols].max() < tolerance).all())


if __name__ == "__main__":
    sh = validate_superheated()
    sat = validate_saturation()

    print(f"Superheated steam: {len(sh)} random (P, T) points, seed {SEED}")
    print(error_table(sh).to_string(index=False, float_format="%.4f"))
    worst = sh.loc[sh[[c for c in sh.columns if c.startswith('err_')]].max(axis=1).idxmax()]
    print(f"Worst point: P = {worst['P_MPa']:.3f} MPa, T = {worst['T_C']:.1f} C")
    print()
    print("Saturation table:")
    print(error_table(sat).to_string(index=False, float_format="%.4f"))
    print()

    ok = passed(sh) and passed(sat)
    print(f"Result: {'PASS' if ok else 'FAIL'} (limit {TOLERANCE_PERCENT} percent)")
    sys.exit(0 if ok else 1)
