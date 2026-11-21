# main.py - run the whole pipeline with ONE command:  python main.py
# Owner: Person C.
# Pipeline: simulate (clean / fault / fault + repair) -> log CSV -> Health Index
#           -> alerts -> text report -> PNG graphs.
# Each module of A and B is imported if it exists; otherwise a MOCK is used and
# the start-up message says so. When the real file is merged, main.py switches
# to it by itself (no edit needed).

import argparse
import warnings

import pandas as pd

from src.dashboard import (plot_before_after_repair, plot_clean_vs_fouled,
                           plot_health, plot_ts_diagram)
from src.logger import log_row, write_report
from src.mock_data import make_mock_df, mock_result
from src.paths import DATA_DIR, OUTPUT_DIR

# ------------------------------------------------------------------ settings
# Baseline plant settings (16.5). Efficiencies and mdot are assumptions.
PARAMS = {"T5": 600, "P5": 15.0, "P6": 1.2, "P7": 0.010,
          "eta_turb": 0.88, "eta_p1": 0.85, "eta_p2": 0.85, "mdot": 100}
SCENARIOS = ["boiler_fouling", "turbine_erosion", "condenser_fouling", "pump_wear"]
DEFAULT_SCENARIO = "turbine_erosion"
DEFAULT_DAYS = 180
DEFAULT_REPAIR_DAY = 120      # assumption for the before/after demo
DEFAULT_NOISE = 0.005
DEFAULT_SEED = 1

# ---------------------------------------------------- real module or MOCK
USING_MOCK = []               # names of modules that fell back to a mock

try:
    from src.simulator import simulate                      # Person A
except ImportError:
    USING_MOCK.append("simulator (A)")

    def simulate(params, scenario="none", days=DEFAULT_DAYS, noise=DEFAULT_NOISE,
                 seed=DEFAULT_SEED, repair_day=None):
        """MOCK of simulator.simulate (same arguments, same columns)."""
        return make_mock_df(days, scenario, repair_day, noise, seed)

try:
    from src.rankine import solve_cycle                     # Person A
except ImportError:
    USING_MOCK.append("rankine (A)")

    def solve_cycle(params, alpha=0.0):
        """MOCK of rankine.solve_cycle: textbook-like result dict."""
        return mock_result()

try:
    from src.health import health_table, remaining_life     # Person B
    from src.monitor import scan                            # Person B
except ImportError:
    USING_MOCK.append("health + monitor (B)")
    remaining_life = None

    # ---- TEMPORARY STUBS: delete this block when B's files are merged ----
    _STUB_DESIGN = {"turbine": 0.88, "pump1": 0.85, "pump2": 0.85,
                    "boiler": 600.0, "condenser": 0.010}      # 16.4
    _STUB_FAIL = {"turbine": 0.70, "pump1": 0.60, "pump2": 0.60,
                  "boiler": 540.0, "condenser": 0.020}        # 16.5
    _STUB_COLUMN = {"turbine": "eta_turb", "pump1": "eta_p1", "pump2": "eta_p2",
                    "boiler": "T5", "condenser": "P7"}
    _STUB_NORMAL, _STUB_WARNING = 80.0, 50.0                  # 16.5

    def health_table(df):
        """MOCK of health.health_table (linear HI between design and fail)."""
        out = pd.DataFrame({"DateTime": df["DateTime"], "day": df["day"]})
        for comp, col in _STUB_COLUMN.items():
            design, fail = _STUB_DESIGN[comp], _STUB_FAIL[comp]
            out[f"HI_{comp}"] = (100 * (df[col] - fail) / (design - fail)).clip(0, 100)
        out["HI_overall"] = out[[f"HI_{c}" for c in _STUB_COLUMN]].min(axis=1)
        return out

    def scan(df):
        """MOCK of monitor.scan: one alert each time a component gets worse."""
        hi_df = health_table(df)
        rows = []
        for comp in _STUB_COLUMN:
            last = "Normal"
            for _, r in hi_df.iterrows():
                hi = r[f"HI_{comp}"]
                status = ("Normal" if hi >= _STUB_NORMAL
                          else "Warning" if hi >= _STUB_WARNING else "Critical")
                if status != last and status != "Normal":
                    msg = f"{comp} health {hi:.0f} is {status} (mock alert)"
                    print(msg)
                    rows.append({"DateTime": r["DateTime"], "component": comp,
                                 "HI": hi, "status": status, "message": msg,
                                 "action": f"Inspect {comp} (mock advice)"})
                last = status
        return pd.DataFrame(rows, columns=["DateTime", "component", "HI",
                                           "status", "message", "action"])
    # ---- end of temporary stubs ----


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with scenario, days, repair_day, seed.
    """
    parser = argparse.ArgumentParser(description="Rankine OFWH health simulation")
    parser.add_argument("--scenario", choices=SCENARIOS, default=DEFAULT_SCENARIO,
                        help="fault scenario to simulate")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help="number of simulated days")
    parser.add_argument("--repair-day", type=int, default=DEFAULT_REPAIR_DAY,
                        help="day of the repair in the before/after run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed (same seed = same result)")
    return parser.parse_args()


def main():
    """Run the full pipeline and write the CSV log, PNG graphs and report."""
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scenario: {args.scenario} | days: {args.days} | repair day: {args.repair_day}")
    if USING_MOCK:
        print("MOCK modules in use:", ", ".join(USING_MOCK))

    # 1) simulate three runs (clean, fault, fault + repair)
    df_clean = simulate(PARAMS, "none", args.days, DEFAULT_NOISE, args.seed)
    df_fault = simulate(PARAMS, args.scenario, args.days, DEFAULT_NOISE, args.seed)
    df_repair = simulate(PARAMS, args.scenario, args.days, DEFAULT_NOISE, args.seed,
                         repair_day=args.repair_day)

    # 2) log the fault run (fresh file each run, so results repeat)
    log_path = DATA_DIR / "log.csv"
    try:
        log_path.unlink(missing_ok=True)
        for _, row in df_fault.iterrows():
            log_row(row.to_dict(), log_path)
    except FileNotFoundError as err:
        print("Could not write the log:", err)

    # 3) Health Index, alerts, remaining life
    hi_df = health_table(df_fault)
    alerts = scan(df_fault)
    if remaining_life is not None:
        days_left = remaining_life(hi_df["day"], hi_df["HI_overall"])
        print("Remaining life (overall HI to Critical):",
              "no decline found" if days_left is None else f"{days_left:.0f} days")

    # 4) text report
    try:
        write_report(df_fault, alerts, OUTPUT_DIR / "report.txt")
    except FileNotFoundError as err:
        print("Could not write the report:", err)

    # 5) graphs
    plot_health(hi_df, OUTPUT_DIR / "health.png")
    plot_clean_vs_fouled(df_clean, df_fault, OUTPUT_DIR / "clean_vs_fouled.png")
    plot_before_after_repair(df_repair, args.repair_day,
                             OUTPUT_DIR / "before_after_repair.png")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # mock-dome warning is noted below
            plot_ts_diagram(solve_cycle(PARAMS), OUTPUT_DIR / "ts_diagram.png")
    except FileNotFoundError as err:
        print("Skipped T-s diagram:", err)

    # 6) summary
    drop = (df_clean["eta_th"].mean() - df_fault["eta_th"].mean()) * 100
    print(f"Mean eta_th: clean {df_clean['eta_th'].mean() * 100:.2f} %, "
          f"fault {df_fault['eta_th'].mean() * 100:.2f} % (loss {drop:.2f} points)")
    print(f"Alerts: {len(alerts)} | files saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
