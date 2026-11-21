"""Time-series plant simulator for the regenerative Rankine cycle with fault injection and repair.

Owner: Person A
"""

import numpy as np
import pandas as pd

from src.degradation import degrade
from src.rankine import solve_cycle


def simulate(params, scenario="none", days=180, noise=0.005, seed=1, repair_day=None):
    """Simulate daily operation of the regenerative Rankine cycle with degradation and sensor noise."""
    if days <= 0:
        raise ValueError("Simulation length 'days' must be a positive integer.")
    if noise < 0.0:
        raise ValueError("Sensor noise standard deviation must be non-negative.")

    random_generator = np.random.default_rng(seed)
    start_timestamp = pd.Timestamp("2026-01-01 00:00")
    records = []

    for current_day in range(days):
        is_repaired = repair_day is not None and current_day >= repair_day
        effective_day = (current_day - repair_day) if is_repaired else current_day

        # Apply physical degradation for the effective operating age
        true_parameters = degrade(params, effective_day, scenario)
        cycle_output = solve_cycle(true_parameters)

        # Apply simulated sensor measurement noise to monitored variables
        def add_sensor_noise(clean_value, upper_bound=None):
            if noise == 0.0:
                return float(clean_value)
            noisy_value = float(clean_value) * (1.0 + random_generator.normal(0.0, noise))
            if upper_bound is not None:
                noisy_value = min(upper_bound, noisy_value)
            return max(1e-6, noisy_value)

        measured_t5 = add_sensor_noise(true_parameters["T5"])
        measured_p7 = add_sensor_noise(true_parameters["P7"])
        measured_eta_turb = add_sensor_noise(true_parameters["eta_turb"], upper_bound=1.0)
        measured_eta_p1 = add_sensor_noise(true_parameters["eta_p1"], upper_bound=1.0)
        measured_eta_p2 = add_sensor_noise(true_parameters["eta_p2"], upper_bound=1.0)
        measured_x7 = add_sensor_noise(cycle_output["x7"], upper_bound=1.0)
        measured_y = add_sensor_noise(cycle_output["y"], upper_bound=1.0)
        measured_w_net = add_sensor_noise(cycle_output["w_net"])
        measured_q_in = add_sensor_noise(cycle_output["q_in"])
        measured_eta_th = add_sensor_noise(cycle_output["eta_th"], upper_bound=1.0)
        measured_power_mw = add_sensor_noise(cycle_output["power_MW"])

        records.append(
            dict(
                DateTime=start_timestamp + pd.Timedelta(days=current_day),
                day=int(current_day),
                T5=measured_t5,
                P7=measured_p7,
                eta_turb=measured_eta_turb,
                eta_p1=measured_eta_p1,
                eta_p2=measured_eta_p2,
                x7=measured_x7,
                y=measured_y,
                w_net=measured_w_net,
                q_in=measured_q_in,
                eta_th=measured_eta_th,
                power_MW=measured_power_mw,
                scenario=scenario,
                repaired=bool(is_repaired),
            )
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    baseline_parameters = dict(
        T5=600.0,
        P5=15.0,
        P6=1.2,
        P7=0.010,
        eta_turb=0.88,
        eta_p1=0.85,
        eta_p2=0.85,
        mdot=100.0,
    )

    print("=== Interactive 180-Day Rankine Cycle Simulator ===")
    while True:
        print("\nScenarios: none, boiler_fouling, turbine_erosion, condenser_fouling, pump_wear")
        chosen_scenario = input("Select scenario (or 'q' to quit): ").strip()
        if chosen_scenario.lower() == "q":
            break

        try:
            total_days = int(input("Number of days to simulate [180]: ") or 180)
            repair_input = input("Repair day (leave blank for no repair): ").strip()
            chosen_repair_day = int(repair_input) if repair_input else None

            simulation_dataframe = simulate(
                baseline_parameters,
                scenario=chosen_scenario,
                days=total_days,
                repair_day=chosen_repair_day,
            )

            print("\n--- First 3 Days ---")
            print(simulation_dataframe.head(3).to_string(index=False))
            print("\n--- Last 3 Days ---")
            print(simulation_dataframe.tail(3).to_string(index=False))
        except ValueError as error_message:
            print("Error:", error_message)
