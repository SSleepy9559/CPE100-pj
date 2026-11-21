# logger.py - write the CSV log and the text report. Owner: Person C.
# log_row   : append one simulation row to a CSV log (data/log.csv).
# write_report : write output/report.txt (first Warning/Critical day, min HI,
#                mean eta_th, recommended action).
# Library functions return nothing and do not print (16.7).

from pathlib import Path

import pandas as pd

# Columns of the Simulation DataFrame, in the agreed order (16.5)
LOG_COLUMNS = ["DateTime", "day", "T5", "P7", "eta_turb", "eta_p1", "eta_p2",
               "x7", "y", "w_net", "q_in", "eta_th", "power_MW",
               "scenario", "repaired"]
DATETIME_FORMAT = "%Y-%m-%d %H:%M"            # CSV format (16.5)
STATUS_ORDER = ("Critical", "Warning")        # most severe first (16.6)
REPORT_WIDTH = 60                             # characters, for the report lines


def log_row(row, path):
    """Append one simulation row to a CSV log file.

    The file is created if it does not exist, and the header is written only
    when the file is new (mode 'a').

    Args:
        row (dict): one row with all keys in LOG_COLUMNS (units as in 16.5).
        path (str or Path): CSV file to append to, e.g. DATA_DIR / 'log.csv'.

    Returns:
        None. The file grows by one line per call.

    Raises:
        ValueError: if the row is not a dict or misses a required column.
        FileNotFoundError: if the file cannot be written (message says where).
    """
    if not isinstance(row, dict):
        raise ValueError("row must be a dict with the Simulation DataFrame columns")
    missing = [col for col in LOG_COLUMNS if col not in row]
    if missing:
        raise ValueError(f"row is missing columns: {missing}")

    path = Path(path)
    data = {col: [row[col]] for col in LOG_COLUMNS}
    data["DateTime"] = [pd.to_datetime(row["DateTime"])]
    frame = pd.DataFrame(data)[LOG_COLUMNS]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        frame.to_csv(path, mode="a", header=is_new, index=False,
                     date_format=DATETIME_FORMAT)
    except FileNotFoundError as err:
        raise FileNotFoundError(f"cannot write log file: {path}") from err


def _first_day(alerts, df, status, component=None):
    """Return a text like '2026-03-02 (day 60)' for the first alert, or 'none'.

    Args:
        alerts (DataFrame): Alerts DataFrame (DateTime, component, HI, status, ...).
        df (DataFrame): Simulation DataFrame, used to look up the day number.
        status (str): 'Warning' or 'Critical'.
        component (str or None): restrict to one component, or None for all.

    Returns:
        str: date and day number of the first matching alert, or 'none'.
    """
    mask = alerts["status"] == status
    if component is not None:
        mask &= alerts["component"] == component
    hits = alerts[mask]
    if hits.empty:
        return "none"
    when = pd.to_datetime(hits["DateTime"]).min()
    day_lookup = dict(zip(pd.to_datetime(df["DateTime"]), df["day"]))
    day_text = f" (day {int(day_lookup[when])})" if when in day_lookup else ""
    return f"{when.strftime(DATETIME_FORMAT)}{day_text}"


def _recommended_action(alerts):
    """Pick the recommended action from the most recent most-severe alert.

    Args:
        alerts (DataFrame): Alerts DataFrame.

    Returns:
        str: action text, or a 'no action' sentence when there are no alerts.
    """
    for status in STATUS_ORDER:
        hits = alerts[alerts["status"] == status]
        if not hits.empty:
            last = hits.sort_values("DateTime").iloc[-1]
            return f"[{status}] {last['component']}: {last['action']}"
    return "No action needed. All components stayed Normal."


def write_report(df, alerts, path):
    """Write a readable text report of one simulation run.

    Contents: run summary, first Warning day, first Critical day (overall and
    per component), minimum HI, mean thermal efficiency, recommended action.

    Args:
        df (DataFrame): Simulation DataFrame (16.5).
        alerts (DataFrame): Alerts DataFrame with DateTime, component, HI,
            status, message, action. It may be empty (no alerts).
        path (str or Path): output file, e.g. OUTPUT_DIR / 'report.txt'.

    Returns:
        None. Writes a text file (efficiencies are shown as percent here).

    Raises:
        ValueError: if df is empty or alerts misses a required column.
        FileNotFoundError: if the file cannot be written (message says where).
    """
    if df is None or len(df) == 0:
        raise ValueError("df is empty: nothing to report")
    needed = ["DateTime", "component", "HI", "status", "message", "action"]
    missing = [col for col in needed if col not in alerts.columns]
    if missing:
        raise ValueError(f"alerts is missing columns: {missing}")

    line = "=" * REPORT_WIDTH
    out = [line, "RANKINE CYCLE (OFWH) - MAINTENANCE REPORT", line]

    # --- run summary
    scenario = df["scenario"].iloc[0] if "scenario" in df.columns else "unknown"
    first = pd.to_datetime(df["DateTime"]).min().strftime(DATETIME_FORMAT)
    last = pd.to_datetime(df["DateTime"]).max().strftime(DATETIME_FORMAT)
    out += [f"Scenario        : {scenario}",
            f"Period          : {first}  to  {last}  ({len(df)} rows)"]
    if "repaired" in df.columns and df["repaired"].any():
        repair_day = int(df.loc[df["repaired"], "day"].min())
        out.append(f"Repair day      : day {repair_day}")

    # --- alerts overview
    out += ["", "ALERTS", "-" * REPORT_WIDTH]
    out.append(f"First Warning   : {_first_day(alerts, df, 'Warning')}")
    out.append(f"First Critical  : {_first_day(alerts, df, 'Critical')}")
    if alerts.empty:
        out.append("Minimum HI      : not below the Normal threshold (no alerts)")
    else:
        worst = alerts.loc[alerts["HI"].idxmin()]
        out.append(f"Minimum HI      : {worst['HI']:.1f} ({worst['component']})")

    for component in sorted(alerts["component"].unique()):
        out.append(f"  {component:<10} Warning: "
                   f"{_first_day(alerts, df, 'Warning', component)}"
                   f" | Critical: {_first_day(alerts, df, 'Critical', component)}")

    # --- performance
    out += ["", "PERFORMANCE", "-" * REPORT_WIDTH,
            f"Mean eta_th     : {df['eta_th'].mean() * 100:.2f} %",
            f"Mean power      : {df['power_MW'].mean():.2f} MW"]
    if "x7" in df.columns:
        out.append(f"Min steam quality x7 : {df['x7'].min():.3f}")

    # --- recommendation
    out += ["", "RECOMMENDED ACTION", "-" * REPORT_WIDTH,
            _recommended_action(alerts), line]

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    except FileNotFoundError as err:
        raise FileNotFoundError(f"cannot write report file: {path}") from err


if __name__ == "__main__":
    # Demo with mock data (B's monitor.scan is not needed yet)
    from src.mock_data import make_mock_df
    from src.paths import DATA_DIR, OUTPUT_DIR

    demo_df = make_mock_df(days=180, scenario="turbine_erosion")
    log_path = DATA_DIR / "log_demo.csv"
    log_path.unlink(missing_ok=True)
    for _, demo_row in demo_df.iterrows():
        log_row(demo_row.to_dict(), log_path)
    print(f"log rows written: {len(pd.read_csv(log_path))}")

    demo_alerts = pd.DataFrame({
        "DateTime": [demo_df["DateTime"][60], demo_df["DateTime"][130]],
        "component": ["turbine", "turbine"],
        "HI": [78.0, 48.5],
        "status": ["Warning", "Critical"],
        "message": ["Turbine health below 80", "Turbine health below 50"],
        "action": ["Schedule blade inspection", "Plan turbine overhaul now"],
    })
    write_report(demo_df, demo_alerts, OUTPUT_DIR / "report_demo.txt")
    print((OUTPUT_DIR / "report_demo.txt").read_text())
