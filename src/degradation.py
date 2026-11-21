"""Equipment degradation models for the regenerative Rankine cycle simulator.

Owner: Person A
"""

VALID_SCENARIOS = [
    "none",
    "boiler_fouling",
    "turbine_erosion",
    "condenser_fouling",
    "pump_wear",
]


def degrade(params, day, scenario):
    """Return a copy of plant parameters updated for equipment degradation at a given day."""
    if day < 0:
        raise ValueError("Simulation day must be non-negative (received day=" + str(day) + ").")
    if scenario not in VALID_SCENARIOS:
        raise ValueError(
            "Unknown degradation scenario '" + str(scenario) + "'. Valid options are: " + str(VALID_SCENARIOS) + "."
        )

    updated_params = params.copy()

    if scenario == "none":
        return updated_params

    if scenario == "boiler_fouling":
        # Boiler outlet temperature drops 0.2 C/day, floored at 540 C
        temperature_drop = 0.2 * day
        updated_params["T5"] = max(540.0, float(params["T5"]) - temperature_drop)

    elif scenario == "turbine_erosion":
        # Turbine isentropic efficiency decays by 0.08% per day
        decay_factor = max(0.01, 1.0 - 0.0008 * day)
        updated_params["eta_turb"] = float(params["eta_turb"]) * decay_factor

    elif scenario == "condenser_fouling":
        # Condenser backpressure rises by 0.4% per day
        growth_factor = 1.0 + 0.004 * day
        updated_params["P7"] = float(params["P7"]) * growth_factor

    elif scenario == "pump_wear":
        # Both feedwater pumps lose 0.1% efficiency per day
        decay_factor = max(0.01, 1.0 - 0.001 * day)
        updated_params["eta_p1"] = float(params["eta_p1"]) * decay_factor
        updated_params["eta_p2"] = float(params["eta_p2"]) * decay_factor

    return updated_params


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

    print("=== Interactive Equipment Degradation Preview ===")
    while True:
        print("\nAvailable scenarios: none, boiler_fouling, turbine_erosion, condenser_fouling, pump_wear")
        chosen_scenario = input("Enter scenario name (or 'q' to quit): ").strip()
        if chosen_scenario.lower() == "q":
            break
        try:
            chosen_day = int(input("Enter simulation day (for example, 90): "))
            degraded = degrade(baseline_parameters, chosen_day, chosen_scenario)
            print("Parameters at day " + str(chosen_day) + " under '" + chosen_scenario + "':")
            print(degraded)
        except ValueError as error_message:
            print("Error:", error_message)
