# dashboard.py - all graphs of the project (PNG files). Owner: Person C.
# plot_health, plot_clean_vs_fouled, plot_before_after_repair, plot_ts_diagram
# Conventions (16.12): matplotlib, figure (8, 5), 150 dpi, same colours everywhere,
# efficiencies shown as percent (x100 only when plotting), title + axis units.
# Library functions return nothing (they save a PNG) and do not print (16.7).

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # save files only, no pop-up window
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

FIG_SIZE = (8, 5)                         # inches (16.12)
DPI = 150                                 # (16.12)

# Component colours (16.12)
COMPONENT_COLORS = {"turbine": "tab:blue", "pump1": "tab:orange",
                    "pump2": "tab:green", "boiler": "tab:red",
                    "condenser": "tab:purple"}
# Status band colours (16.12)
BAND_COLORS = {"Normal": "tab:green", "Warning": "tab:orange", "Critical": "tab:red"}

# Health Index bands (16.5, owned by B; defaults here, can be passed in)
NORMAL_THRESHOLD = 80.0
WARNING_THRESHOLD = 50.0

CLEAN_COLOR = "tab:green"
FAULT_COLOR = "tab:red"
EXPANSION_COLOR = "crimson"               # expansion 5-6-7 (clearly different)
PROCESS_COLOR = "tab:blue"                # heating and pumping lines
BAND_ALPHA = 0.12
MIN_SAT_COLUMNS = ["Tsat", "sf", "sg"]
# Label offsets (points) so that close states (1-2 and 3-4) do not overlap
LABEL_OFFSETS = {1: (-16, -14), 2: (6, 6), 3: (-18, 8), 4: (8, -14)}
DEFAULT_LABEL_OFFSET = (6, 6)
X7_LABEL_OFFSET = (8, -14)
N_BOILER_POINTS = 25                      # points used to draw the heating line 4-5


def _require_columns(df, columns, name):
    """Raise ValueError if a DataFrame misses required columns.

    Args:
        df (DataFrame): table to check.
        columns (list of str): column names that must exist.
        name (str): name of the table for the error message.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _save(fig, path):
    """Save a figure as PNG (150 dpi) and close it.

    Args:
        fig (Figure): matplotlib figure.
        path (str or Path): output PNG path.

    Raises:
        FileNotFoundError: if the file cannot be written.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    except FileNotFoundError as err:
        raise FileNotFoundError(f"cannot save figure: {path}") from err
    finally:
        plt.close(fig)


def plot_health(df_hi, path, normal=NORMAL_THRESHOLD, warning=WARNING_THRESHOLD):
    """Plot the Health Index (0-100) of every component against time.

    Args:
        df_hi (DataFrame): Health DataFrame with day, HI_turbine, HI_pump1,
            HI_pump2, HI_boiler, HI_condenser, HI_overall (floats 0-100).
        path (str or Path): output PNG.
        normal (float): lower edge of the Normal band (HI).
        warning (float): lower edge of the Warning band (HI).

    Returns:
        None. Saves a PNG.
    """
    cols = ["day"] + [f"HI_{c}" for c in COMPONENT_COLORS] + ["HI_overall"]
    _require_columns(df_hi, cols, "df_hi")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.axhspan(normal, 100, color=BAND_COLORS["Normal"], alpha=BAND_ALPHA, label="Normal")
    ax.axhspan(warning, normal, color=BAND_COLORS["Warning"], alpha=BAND_ALPHA, label="Warning")
    ax.axhspan(0, warning, color=BAND_COLORS["Critical"], alpha=BAND_ALPHA, label="Critical")
    for comp, color in COMPONENT_COLORS.items():
        ax.plot(df_hi["day"], df_hi[f"HI_{comp}"], color=color, label=comp)
    ax.plot(df_hi["day"], df_hi["HI_overall"], color="black", linestyle="--",
            linewidth=2, label="overall")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Health Index (0-100)")
    ax.set_title("Equipment Health Index")
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_clean_vs_fouled(df_clean, df_fault, path):
    """Compare a clean run with a faulty (fouled/worn) run.

    Top: thermal efficiency (%). Bottom: steam quality x7 at the turbine exit.

    Args:
        df_clean (DataFrame): Simulation DataFrame, scenario 'none'.
        df_fault (DataFrame): Simulation DataFrame with a fault scenario.
        path (str or Path): output PNG.

    Returns:
        None. Saves a PNG.
    """
    cols = ["day", "eta_th", "x7", "scenario"]
    _require_columns(df_clean, cols, "df_clean")
    _require_columns(df_fault, cols, "df_fault")
    fault_name = df_fault["scenario"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    for ax, col, scale, label in ((ax1, "eta_th", 100, "Thermal efficiency (%)"),
                                  (ax2, "x7", 1, "Steam quality x7 (fraction)")):
        ax.plot(df_clean["day"], df_clean[col] * scale, color=CLEAN_COLOR, label="clean")
        ax.plot(df_fault["day"], df_fault[col] * scale, color=FAULT_COLOR,
                label=f"fault: {fault_name}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    ax2.set_xlabel("Time (days)")
    ax1.set_title(f"Clean vs {fault_name}")
    fig.tight_layout()
    _save(fig, path)


def plot_before_after_repair(df, repair_day, path):
    """Show efficiency and power before and after a repair.

    Top: thermal efficiency (%). Bottom: net power (MW). A vertical dashed
    line labelled 'repair' marks the repair day.

    Args:
        df (DataFrame): Simulation DataFrame that includes the repair.
        repair_day (int): day of the repair (days).
        path (str or Path): output PNG.

    Returns:
        None. Saves a PNG.

    Raises:
        ValueError: if repair_day is outside the days in df.
    """
    _require_columns(df, ["day", "eta_th", "power_MW"], "df")
    if not df["day"].min() <= repair_day <= df["day"].max():
        raise ValueError("repair_day must be inside the range of df['day']")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    for ax, col, scale, label in ((ax1, "eta_th", 100, "Thermal efficiency (%)"),
                                  (ax2, "power_MW", 1, "Net power (MW)")):
        ax.plot(df["day"], df[col] * scale, color=COMPONENT_COLORS["turbine"], label=col)
        ax.axvline(repair_day, color="black", linestyle="--", label="repair")
        ax.annotate("repair", (repair_day, ax.get_ylim()[0]), textcoords="offset points",
                    xytext=(4, 6), fontsize=9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=8)
    ax2.set_xlabel("Time (days)")
    ax1.set_title("Before and after repair")
    fig.tight_layout()
    _save(fig, path)


def _load_sat_dome(sat_table):
    """Return the saturation table used to draw the T-s dome.

    Args:
        sat_table (DataFrame or None): table with Tsat, sf, sg; if None, read
            data/sat_table.csv (made by Person A).

    Returns:
        DataFrame with Tsat (C), sf and sg (kJ/kg.K), sorted by Tsat.
        If the CSV does not exist, an approximate mock dome is used with a
        warning (needs mock_sat_table in mock_data.py).

    Raises:
        FileNotFoundError: if the CSV and mock_sat_table are both missing.
    """
    if sat_table is None:
        from src.paths import DATA_DIR
        csv_path = DATA_DIR / "sat_table.csv"
        try:
            sat_table = pd.read_csv(csv_path)
        except FileNotFoundError as err:
            try:
                from src.mock_data import mock_sat_table
            except ImportError:
                raise FileNotFoundError(
                    f"{csv_path} not found: ask Person A to commit "
                    "data/sat_table.csv (made by scripts/make_steam_table.py), "
                    "then run git pull origin main") from err
            warnings.warn(f"{csv_path} not found: using an approximate MOCK dome",
                          stacklevel=3)
            sat_table = mock_sat_table()
    _require_columns(sat_table, MIN_SAT_COLUMNS, "sat_table")
    return sat_table.sort_values("Tsat")


def _boiler_path(states, sat):
    """Return the heating path 4 -> 5 as lists of (s, T).

    If the table has a pressure column P, the path follows the saturated liquid
    line up to Tsat(P5), then the vapour line, then superheats to state 5.
    Otherwise it is a straight line from state 4 to state 5 (simplified).

    Args:
        states (dict): result['states'] (needs P, T, s of states 4 and 5).
        sat (DataFrame): saturation table sorted by Tsat.

    Returns:
        tuple (list of s, list of T): entropy (kJ/(kg.K)) and temperature (C).
    """
    s4, t4 = states[4]["s"], states[4]["T"]
    s5, t5 = states[5]["s"], states[5]["T"]
    if "P" not in sat.columns or "P" not in states[5]:
        return [s4, s5], [t4, t5]
    t_sat = float(np.interp(states[5]["P"], sat["P"], sat["Tsat"]))
    if t_sat <= t4 or t_sat >= t5:
        return [s4, s5], [t4, t5]
    liquid_t = np.linspace(t4, t_sat, N_BOILER_POINTS)
    liquid_s = np.interp(liquid_t, sat["Tsat"], sat["sf"])
    s_vap = float(np.interp(t_sat, sat["Tsat"], sat["sg"]))
    return (list(liquid_s) + [s_vap, s5], list(liquid_t) + [t_sat, t5])


def plot_ts_diagram(result, path, sat_table=None):
    """Draw the T-s diagram: saturation dome, states 1-7 and the cycle lines.

    Expansion 5-6-7 is drawn in a different colour from heating and pumping.
    The OFWH line 2-3 is a straight line between the states (simplified). The
    boiler line 4-5 follows the dome if the table has a pressure column P.

    Args:
        result (dict): result of solve_cycle (or mock_result) with
            result['states'][n] = {'T': C, 's': kJ/(kg.K), ...} for n = 1..7.
        path (str or Path): output PNG.
        sat_table (DataFrame or None): saturation table (Tsat, sf, sg). None
            means read data/sat_table.csv.

    Returns:
        None. Saves a PNG.
    """
    states = result["states"]
    missing = [n for n in range(1, 8) if n not in states]
    if missing:
        raise ValueError(f"result['states'] is missing states: {missing}")
    sat = _load_sat_dome(sat_table)

    s = {n: states[n]["s"] for n in range(1, 8)}
    t = {n: states[n]["T"] for n in range(1, 8)}

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # saturation dome: liquid line up, vapour line down
    dome_s = list(sat["sf"]) + list(sat["sg"])[::-1]
    dome_t = list(sat["Tsat"]) + list(sat["Tsat"])[::-1]
    ax.fill(dome_s, dome_t, color="lightsteelblue", alpha=0.4)
    ax.plot(dome_s, dome_t, color="steelblue", linewidth=1.2, label="saturation dome")

    # heating and pumping lines
    process = [(1, 2), (2, 3), (3, 4), (7, 1)]
    for i, (a, b) in enumerate(process):
        ax.plot([s[a], s[b]], [t[a], t[b]], color=PROCESS_COLOR, linewidth=1.8,
                label="heating / pumping / condensing" if i == 0 else None)
    boiler_s, boiler_t = _boiler_path(states, sat)
    ax.plot(boiler_s, boiler_t, color=PROCESS_COLOR, linewidth=1.8)
    # expansion 5 -> 6 -> 7
    ax.plot([s[5], s[6], s[7]], [t[5], t[6], t[7]], color=EXPANSION_COLOR,
            linewidth=2.5, label="expansion 5-6-7")

    for n in range(1, 8):
        ax.plot(s[n], t[n], "o", color="black", markersize=5)
        ax.annotate(str(n), (s[n], t[n]), textcoords="offset points",
                    xytext=LABEL_OFFSETS.get(n, DEFAULT_LABEL_OFFSET),
                    fontsize=10, fontweight="bold")
    x7 = states[7].get("x")
    if x7 is not None:
        ax.annotate(f"x7 = {x7:.3f}", (s[7], t[7]), textcoords="offset points",
                    xytext=X7_LABEL_OFFSET, fontsize=9, color=EXPANSION_COLOR)

    ax.set_xlabel("Entropy s (kJ/(kg.K))")
    ax.set_ylabel("Temperature T (C)")
    ax.set_title("T-s diagram: regenerative Rankine cycle with open FWH")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, path)


if __name__ == "__main__":
    # Demo with mock data. Run from the project root: python -m src.dashboard
    import numpy as np
    from src.mock_data import make_mock_df, mock_result
    from src.paths import OUTPUT_DIR

    def demo_health_table(sim):
        """Mock Health table (B's health_table is not ready yet)."""
        def hi(meas, design, fail):
            return 100 * np.clip((meas - fail) / (design - fail), 0, 1)
        out = pd.DataFrame({"DateTime": sim["DateTime"], "day": sim["day"]})
        out["HI_turbine"] = hi(sim["eta_turb"], 0.88, 0.70)
        out["HI_pump1"] = hi(sim["eta_p1"], 0.85, 0.60)
        out["HI_pump2"] = hi(sim["eta_p2"], 0.85, 0.60)
        out["HI_boiler"] = hi(sim["T5"], 600, 540)
        out["HI_condenser"] = hi(sim["P7"], 0.010, 0.020)   # higher is worse
        comp_cols = [c for c in out.columns if c.startswith("HI_")]
        out["HI_overall"] = out[comp_cols].min(axis=1)
        return out

    clean = make_mock_df(180, scenario="none")
    fault = make_mock_df(180, scenario="turbine_erosion")
    repaired = make_mock_df(180, scenario="turbine_erosion", repair_day=120)
    plot_health(demo_health_table(fault), OUTPUT_DIR / "health.png")
    plot_clean_vs_fouled(clean, fault, OUTPUT_DIR / "clean_vs_fouled.png")
    plot_before_after_repair(repaired, 120, OUTPUT_DIR / "before_after_repair.png")
    plot_ts_diagram(mock_result(), OUTPUT_DIR / "ts_diagram.png")
    print("saved 4 PNG files in", OUTPUT_DIR)
