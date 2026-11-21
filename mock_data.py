# mock_data.py
# Fake (mock) data for the CPE 100 Rankine-OFWH project. Owner: Person C.
# Purpose: let B, C and D work on day 1 without waiting for A's real solver.
# Shapes (columns, keys, names) follow sections 16.5 - 16.7 of the project doc.
# All numbers here are ASSUMED / textbook-like, NOT computed from a steam table.
# Replace with real data from rankine.py / simulator.py at Milestone 1 and 2.

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
START_DATE = "2026-01-01 00:00"      # same start date for all runs (16.5)
DEFAULT_DAYS = 180                   # simulation length (16.5)
DEFAULT_NOISE = 0.005                # 0.5 percent noise (16.5)
DEFAULT_SEED = 1                     # shared random seed (16.5)

SCENARIOS = ("none", "boiler_fouling", "turbine_erosion",
             "condenser_fouling", "pump_wear")   # exact names (16.6)

# Design values (16.4)
T5_DESIGN = 600.0        # C
P7_DESIGN = 0.010        # MPa
ETA_TURB_DESIGN = 0.88   # assumption
ETA_PUMP_DESIGN = 0.85   # assumption
MDOT = 100.0             # kg/s, assumption

# Fault rates (16.4, demo values)
T5_DROP_PER_DAY = 0.2    # C/day
T5_FLOOR = 540.0         # C
TURB_LOSS_PER_DAY = 0.0008
P7_GROWTH_PER_DAY = 0.004
PUMP_LOSS_PER_DAY = 0.001

# Mock "real cycle" baseline (assumptions, roughly right for eta_turb = 0.88)
Y_DESIGN = 0.227         # extraction fraction
X7_DESIGN = 0.85         # quality at turbine exit (design x7 is 0.80-0.85)
Q_IN_DESIGN = 2769.0     # kJ/kg, h5 - h4 (textbook-like)
ETA_TH_DESIGN = 0.43     # real-cycle efficiency with eta_turb = 0.88 (assumption)

# Simple linear sensitivities used only to make mock curves look plausible
K_ETA_TURB = 0.50        # d(eta_th)/d(eta_turb)
K_T5 = 0.0003            # d(eta_th)/d(T5) per C
K_P7 = -1.7              # d(eta_th)/d(P7) per MPa
K_PUMP = 0.02            # d(eta_th)/d(eta_pump)
K_X7_ETA = 0.6           # lower eta_turb -> wetter? no: drier exhaust (physics)
K_X7_T5 = 0.0015         # lower T5 -> lower x7 per C
K_X7_P7 = -1.5           # higher P7 -> lower x7 per MPa


def _check_inputs(days, scenario, repair_day):
    """Raise ValueError for invalid arguments."""
    if days <= 0:
        raise ValueError("days must be a positive integer")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got '{scenario}'")
    if repair_day is not None and not 0 <= repair_day <= days:
        raise ValueError("repair_day must be between 0 and days")


def make_mock_df(days=DEFAULT_DAYS, scenario="turbine_erosion", repair_day=None,
                 noise=DEFAULT_NOISE, seed=DEFAULT_SEED):
    """Make a fake Simulation DataFrame (linear decay plus noise), one row per day.

    The first argument matches the contract make_mock_df(days=180). The other
    arguments are optional extras (defaults keep the contract behaviour).

    Args:
        days (int): number of days (rows).
        scenario (str): 'none', 'boiler_fouling', 'turbine_erosion',
            'condenser_fouling' or 'pump_wear'.
        repair_day (int or None): day when the faulty part is reset to design.
        noise (float): relative noise on measured values (0.005 = 0.5 percent).
        seed (int): random seed, so results repeat.

    Returns:
        pandas.DataFrame with columns DateTime, day, T5, P7, eta_turb, eta_p1,
        eta_p2, x7, y, w_net, q_in, eta_th, power_MW, scenario, repaired.
        Units: T5 in C, P7 in MPa, efficiencies/x7/y as fractions,
        w_net and q_in in kJ/kg, power_MW in MW.
    """
    _check_inputs(days, scenario, repair_day)
    rng = np.random.default_rng(seed)

    day = np.arange(days)
    repaired = np.zeros(days, dtype=bool) if repair_day is None else day >= repair_day
    age = np.where(repaired, 0, day)      # a repaired part is back to design

    # True (noise-free) values, start at design
    t5 = np.full(days, T5_DESIGN)
    p7 = np.full(days, P7_DESIGN)
    eta_turb = np.full(days, ETA_TURB_DESIGN)
    eta_p1 = np.full(days, ETA_PUMP_DESIGN)
    eta_p2 = np.full(days, ETA_PUMP_DESIGN)

    # Each scenario changes only its own variables (16.4)
    if scenario == "boiler_fouling":
        t5 = np.maximum(T5_FLOOR, T5_DESIGN - T5_DROP_PER_DAY * age)
    elif scenario == "turbine_erosion":
        eta_turb = ETA_TURB_DESIGN * (1 - TURB_LOSS_PER_DAY * age)
    elif scenario == "condenser_fouling":
        p7 = P7_DESIGN * (1 + P7_GROWTH_PER_DAY * age)
    elif scenario == "pump_wear":
        eta_p1 = ETA_PUMP_DESIGN * (1 - PUMP_LOSS_PER_DAY * age)
        eta_p2 = ETA_PUMP_DESIGN * (1 - PUMP_LOSS_PER_DAY * age)

    # Derived quantities (mock linear model around the design point)
    eta_th = (ETA_TH_DESIGN
              + K_ETA_TURB * (eta_turb - ETA_TURB_DESIGN)
              + K_T5 * (t5 - T5_DESIGN)
              + K_P7 * (p7 - P7_DESIGN)
              + K_PUMP * ((eta_p1 - ETA_PUMP_DESIGN) + (eta_p2 - ETA_PUMP_DESIGN)))
    x7 = (X7_DESIGN
          + K_X7_ETA * (ETA_TURB_DESIGN - eta_turb)
          + K_X7_T5 * (t5 - T5_DESIGN)
          + K_X7_P7 * (p7 - P7_DESIGN))
    y = Y_DESIGN - 0.1 * (ETA_TURB_DESIGN - eta_turb)
    q_in = np.full(days, Q_IN_DESIGN)
    w_net = eta_th * q_in

    def noisy(values):
        """Return values with relative Gaussian noise (sensor-like)."""
        return values * (1 + rng.normal(0.0, noise, size=days))

    eta_th_m = noisy(eta_th)
    q_in_m = noisy(q_in)
    w_net_m = noisy(w_net)

    df = pd.DataFrame({
        "DateTime": pd.date_range(START_DATE, periods=days, freq="D"),
        "day": day,
        "T5": noisy(t5),
        "P7": noisy(p7),
        "eta_turb": noisy(eta_turb),
        "eta_p1": noisy(eta_p1),
        "eta_p2": noisy(eta_p2),
        "x7": np.clip(noisy(x7), 0.0, 1.0),
        "y": noisy(y),
        "w_net": w_net_m,
        "q_in": q_in_m,
        "eta_th": eta_th_m,
        "power_MW": w_net_m * MDOT / 1000.0,
        "scenario": scenario,
        "repaired": repaired,
    })
    return df


def mock_result():
    """Make a fake solve_cycle() result with textbook-like ideal-cycle values.

    Values are approximate (Cengel-style, 15 MPa/600 C/1.2 MPa/10 kPa) and are
    NOT from our own steam table. Work and heat are kJ per kg of boiler steam.

    Returns:
        dict with keys 'states' (int keys 1-7, each with P, T, h, s, x),
        'y', 'w_turb', 'w_pump', 'w_net', 'q_in', 'q_out', 'eta_th',
        'x7', 'power_MW'.
    """
    # (P [MPa], T [C], h [kJ/kg], s [kJ/kg.K], x [fraction])
    raw = {
        1: (0.010, 45.81, 191.81, 0.6492, 0.0),
        2: (1.2, 45.9, 193.0, 0.6493, 0.0),
        3: (1.2, 187.96, 798.33, 2.2159, 0.0),
        4: (15.0, 190.0, 814.0, 2.2200, 0.0),
        5: (15.0, 600.0, 3583.1, 6.6796, 1.0),
        6: (1.2, 218.5, 2860.2, 6.6796, 1.0),
        7: (0.010, 45.81, 2115.3, 6.6796, 0.804),
    }
    states = {k: {"P": p, "T": t, "h": h, "s": s, "x": x}
              for k, (p, t, h, s, x) in raw.items()}

    h = {k: v["h"] for k, v in states.items()}
    y = (h[3] - h[2]) / (h[6] - h[2])                       # FWH balance
    w_pump1 = 0.00101 * (1.2 - 0.010) * 1000                # v1 * dP (kPa)
    w_pump2 = 0.001138 * (15.0 - 1.2) * 1000                # v3 * dP (kPa)
    w_pump = (1 - y) * w_pump1 + w_pump2
    w_turb = (h[5] - h[6]) + (1 - y) * (h[6] - h[7])
    q_in = h[5] - h[4]
    q_out = (1 - y) * (h[7] - h[1])
    w_net = w_turb - w_pump

    return {
        "states": states,
        "y": y,
        "w_turb": w_turb,
        "w_pump": w_pump,
        "w_net": w_net,
        "q_in": q_in,
        "q_out": q_out,
        "eta_th": w_net / q_in,
        "x7": states[7]["x"],
        "power_MW": w_net * MDOT / 1000.0,
    }


if __name__ == "__main__":
    # Small demo: print the mock results and a few rows
    res = mock_result()
    print(f"mock_result: y={res['y']:.3f}  x7={res['x7']:.3f}  "
          f"eta_th={res['eta_th']:.3f}  w_net={res['w_net']:.1f} kJ/kg")
    demo = make_mock_df(days=180, scenario="turbine_erosion", repair_day=120)
    print(demo[["DateTime", "day", "eta_turb", "x7", "eta_th", "repaired"]]
          .iloc[[0, 60, 119, 120, 179]].to_string(index=False))
