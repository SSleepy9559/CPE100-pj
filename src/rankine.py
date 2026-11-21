"""Seven-state regenerative Rankine cycle solver with an open feedwater heater.

Owner: Person A
"""

from src.steam_props import props_Ph, props_Ps, props_PT, sat_props


def solve_cycle(params, alpha=0.0):
    """Solve all seven state points and performance metrics of the regenerative Rankine cycle."""
    boiler_pressure = float(params["P5"])
    boiler_temperature = float(params["T5"])
    extraction_pressure = float(params["P6"])
    condenser_pressure = float(params["P7"])
    turbine_efficiency = float(params["eta_turb"])
    pump1_efficiency = float(params["eta_p1"])
    pump2_efficiency = float(params["eta_p2"])
    mass_flow_rate = float(params["mdot"])

    if not (condenser_pressure < extraction_pressure < boiler_pressure):
        raise ValueError(
            "Pressures must satisfy P7 < P6 < P5 (received P7="
            + str(condenser_pressure) + ", P6=" + str(extraction_pressure)
            + ", P5=" + str(boiler_pressure) + " MPa)."
        )
    if not (0.0 < turbine_efficiency <= 1.0 and 0.0 < pump1_efficiency <= 1.0 and 0.0 < pump2_efficiency <= 1.0):
        raise ValueError("Component efficiencies must be fractions strictly greater than 0 and at most 1.")
    if mass_flow_rate <= 0.0:
        raise ValueError("Boiler steam mass flow rate 'mdot' must be positive.")

    # State 1: Saturated liquid leaving the condenser at P7
    saturated_state1 = sat_props(condenser_pressure)
    state1 = dict(
        P=condenser_pressure,
        T=saturated_state1["Tsat"],
        h=saturated_state1["hf"],
        s=saturated_state1["sf"],
        x=0.0,
    )

    # State 2: Compressed liquid after Pump I at P6 (multiply MPa difference by 1000 for kJ/kg)
    pump1_work = saturated_state1["vf"] * (extraction_pressure - condenser_pressure) * 1000.0 / pump1_efficiency
    enthalpy2 = state1["h"] + pump1_work
    state2 = dict(
        P=extraction_pressure,
        T=state1["T"],
        h=float(enthalpy2),
        s=state1["s"],
        x=0.0,
    )

    # State 3: Saturated liquid leaving the open feedwater heater at P6
    saturated_state3 = sat_props(extraction_pressure)
    state3 = dict(
        P=extraction_pressure,
        T=saturated_state3["Tsat"],
        h=saturated_state3["hf"],
        s=saturated_state3["sf"],
        x=0.0,
    )

    # State 4: Compressed liquid after Pump II at P5
    pump2_work = saturated_state3["vf"] * (boiler_pressure - extraction_pressure) * 1000.0 / pump2_efficiency
    enthalpy4 = state3["h"] + pump2_work
    state4 = dict(
        P=boiler_pressure,
        T=state3["T"],
        h=float(enthalpy4),
        s=state3["s"],
        x=0.0,
    )

    # State 5: Superheated steam leaving the boiler at P5 and T5
    superheated_state5 = props_PT(boiler_pressure, boiler_temperature)
    state5 = dict(
        P=boiler_pressure,
        T=boiler_temperature,
        h=superheated_state5["h"],
        s=superheated_state5["s"],
        x=1.0,
    )

    # State 6: Turbine extraction at P6 (first stage expansion from State 5)
    isentropic_state6 = props_Ps(extraction_pressure, state5["s"])
    stage1_efficiency = turbine_efficiency
    if alpha > 0.0 and isentropic_state6["x"] < 1.0:
        average_quality_stage1 = 0.5 * (1.0 + isentropic_state6["x"])
        stage1_efficiency = turbine_efficiency * (1.0 - alpha * (1.0 - average_quality_stage1))

    enthalpy6 = state5["h"] - stage1_efficiency * (state5["h"] - isentropic_state6["h"])
    actual_state6 = props_Ph(extraction_pressure, enthalpy6)
    state6 = dict(
        P=extraction_pressure,
        T=actual_state6["T"],
        h=float(enthalpy6),
        s=actual_state6["s"],
        x=actual_state6["x"],
    )

    # State 7: Turbine exhaust at P7 (second stage expansion starting from actual State 6)
    isentropic_state7 = props_Ps(condenser_pressure, state6["s"])
    stage2_efficiency = turbine_efficiency
    if alpha > 0.0 and isentropic_state7["x"] < 1.0:
        starting_quality = state6["x"] if state6["x"] < 1.0 else 1.0
        average_quality_stage2 = 0.5 * (starting_quality + isentropic_state7["x"])
        stage2_efficiency = turbine_efficiency * (1.0 - alpha * (1.0 - average_quality_stage2))

    enthalpy7 = state6["h"] - stage2_efficiency * (state6["h"] - isentropic_state7["h"])
    actual_state7 = props_Ph(condenser_pressure, enthalpy7)
    state7 = dict(
        P=condenser_pressure,
        T=actual_state7["T"],
        h=float(enthalpy7),
        s=actual_state7["s"],
        x=actual_state7["x"],
    )

    # Open feedwater heater adiabatic energy balance
    extraction_fraction = (state3["h"] - state2["h"]) / (state6["h"] - state2["h"])

    # Work and heat per kilogram of steam passing through the boiler
    turbine_work = (state5["h"] - state6["h"]) + (1.0 - extraction_fraction) * (state6["h"] - state7["h"])
    total_pump_work = (1.0 - extraction_fraction) * pump1_work + pump2_work
    net_work = turbine_work - total_pump_work
    heat_added = state5["h"] - state4["h"]
    heat_rejected = (1.0 - extraction_fraction) * (state7["h"] - state1["h"])
    thermal_efficiency = net_work / heat_added
    net_power_mw = net_work * mass_flow_rate / 1000.0

    states_dictionary = dict()
    states_dictionary[1] = state1
    states_dictionary[2] = state2
    states_dictionary[3] = state3
    states_dictionary[4] = state4
    states_dictionary[5] = state5
    states_dictionary[6] = state6
    states_dictionary[7] = state7

    return dict(
        states=states_dictionary,
        y=float(extraction_fraction),
        w_turb=float(turbine_work),
        w_pump=float(total_pump_work),
        w_net=float(net_work),
        q_in=float(heat_added),
        q_out=float(heat_rejected),
        eta_th=float(thermal_efficiency),
        x7=float(state7["x"]),
        power_MW=float(net_power_mw),
    )


if __name__ == "__main__":
    default_parameters = dict(
        T5=600.0,
        P5=15.0,
        P6=1.2,
        P7=0.010,
        eta_turb=0.88,
        eta_p1=0.85,
        eta_p2=0.85,
        mdot=100.0,
    )

    print("=== Interactive Regenerative Rankine Cycle Calculator ===")
    while True:
        print("\n1: Run baseline design cycle (15 MPa, 600 C, P6=1.2 MPa, P7=0.010 MPa)")
        print("2: Run ideal cycle validation (all efficiencies = 1.0)")
        print("3: Enter custom cycle parameters")
        print("q: Quit")
        choice = input("Select option: ").strip().lower()

        if choice == "q":
            break

        try:
            active_parameters = default_parameters.copy()
            moisture_coefficient = 0.0

            if choice == "2":
                active_parameters["eta_turb"] = 1.0
                active_parameters["eta_p1"] = 1.0
                active_parameters["eta_p2"] = 1.0
            elif choice == "3":
                active_parameters["P5"] = float(input("Boiler pressure P5 (MPa) [15.0]: ") or 15.0)
                active_parameters["T5"] = float(input("Boiler temperature T5 (degrees C) [600.0]: ") or 600.0)
                active_parameters["P6"] = float(input("Extraction pressure P6 (MPa) [1.2]: ") or 1.2)
                active_parameters["P7"] = float(input("Condenser pressure P7 (MPa) [0.010]: ") or 0.010)
                active_parameters["eta_turb"] = float(input("Turbine efficiency (0 to 1) [0.88]: ") or 0.88)
                active_parameters["eta_p1"] = float(input("Pump I efficiency (0 to 1) [0.85]: ") or 0.85)
                active_parameters["eta_p2"] = float(input("Pump II efficiency (0 to 1) [0.85]: ") or 0.85)
                active_parameters["mdot"] = float(input("Boiler mass flow rate (kg/s) [100.0]: ") or 100.0)
                moisture_coefficient = float(input("Baumann moisture factor alpha [0.0]: ") or 0.0)
            elif choice != "1":
                print("Invalid option.")
                continue

            cycle_result = solve_cycle(active_parameters, alpha=moisture_coefficient)
            print("\n--- State Points ---")
            for state_number, properties in cycle_result["states"].items():
                print(
                    "State " + str(state_number) + ":",
                    "P=" + "%.3f" % properties["P"] + " MPa,",
                    "T=" + "%.2f" % properties["T"] + " C,",
                    "h=" + "%.2f" % properties["h"] + " kJ/kg,",
                    "s=" + "%.4f" % properties["s"] + " kJ/(kg.K),",
                    "x=" + "%.4f" % properties["x"],
                )

            print("\n--- Cycle Summary ---")
            print("Extraction fraction (y):   ", "%.4f" % cycle_result["y"])
            print("Turbine exhaust quality:   ", "%.4f" % cycle_result["x7"])
            print("Net work output (w_net):   ", "%.2f" % cycle_result["w_net"], "kJ/kg")
            print("Heat added (q_in):         ", "%.2f" % cycle_result["q_in"], "kJ/kg")
            print("Thermal efficiency:        ", "%.2f" % (cycle_result["eta_th"] * 100.0), "%")
            print("Net power output:          ", "%.2f" % cycle_result["power_MW"], "MW")
        except ValueError as error_message:
            print("Error:", error_message)
