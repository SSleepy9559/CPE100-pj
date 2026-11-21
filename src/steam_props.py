"""Steam property lookup and interpolation module for the Rankine Open Feedwater Heater simulator.

Owner: Person A
"""

import numpy as np
import pandas as pd

# 1. Read sat_table.csv and sort it by pressure "P"
saturated_table = pd.read_csv("data/sat_table.csv")
saturated_table.columns = saturated_table.columns.str.strip()
saturated_table = saturated_table.sort_values("P").reset_index(drop=True)

# 2. Read sh_table.csv and filter so only vapor rows are kept
superheated_raw = pd.read_csv("data/sh_table.csv")
superheated_raw.columns = superheated_raw.columns.str.strip()
superheated_vapor = superheated_raw[
    superheated_raw["phase"].astype(str).str.strip() == "vapor"
].copy()

numeric_columns = ["P", "T", "v", "h", "s"]
for column_name in numeric_columns:
    superheated_vapor[column_name] = pd.to_numeric(superheated_vapor[column_name], errors="coerce")
superheated_vapor = superheated_vapor.dropna(subset=numeric_columns).reset_index(drop=True)


def _interpolate_superheated_2d(P, T=None, s=None, h=None):
    """Interpolate superheated vapor properties at pressure P and one second property."""
    pressures = np.sort(superheated_vapor["P"].unique())
    if P < pressures[0] or P > pressures[-1]:
        raise ValueError(
            "Pressure " + str(P) + " MPa is outside the superheated steam table range ["
            + str(pressures[0]) + ", " + str(pressures[-1]) + "] MPa."
        )

    index = int(np.searchsorted(pressures, P))
    if index < len(pressures) and np.isclose(pressures[index], P):
        pressure_levels = [pressures[index]]
    else:
        pressure_levels = [pressures[index - 1], pressures[index]]

    results_at_pressure = []
    for pressure_level in pressure_levels:
        subset = superheated_vapor[superheated_vapor["P"] == pressure_level]
        if T is not None:
            subset = subset.sort_values("T")
            independent_array = subset["T"].to_numpy()
            target_value = T
            property_label = "Temperature (°C)"
        elif s is not None:
            subset = subset.sort_values("s")
            independent_array = subset["s"].to_numpy()
            target_value = s
            property_label = "Specific Entropy [kJ/(kg K)]"
        else:
            subset = subset.sort_values("h")
            independent_array = subset["h"].to_numpy()
            target_value = h
            property_label = "Specific Enthalpy (kJ/kg)"

        if target_value < independent_array[0] or target_value > independent_array[-1]:
            raise ValueError(
                property_label + " " + str(target_value) + " is outside the steam table range ["
                + str(independent_array[0]) + ", " + str(independent_array[-1])
                + "] at pressure " + str(pressure_level) + " MPa."
            )

        temperature_value = float(np.interp(target_value, independent_array, subset["T"]))
        enthalpy_value = float(np.interp(target_value, independent_array, subset["h"]))
        entropy_value = float(np.interp(target_value, independent_array, subset["s"]))
        volume_value = float(np.interp(target_value, independent_array, subset["v"]))
        results_at_pressure.append((temperature_value, enthalpy_value, entropy_value, volume_value))

    if len(pressure_levels) == 1:
        temperature_out, enthalpy_out, entropy_out, volume_out = results_at_pressure[0]
    else:
        weight = (P - pressure_levels[0]) / (pressure_levels[1] - pressure_levels[0])
        temperature_out = results_at_pressure[0][0] + weight * (results_at_pressure[1][0] - results_at_pressure[0][0])
        enthalpy_out = results_at_pressure[0][1] + weight * (results_at_pressure[1][1] - results_at_pressure[0][1])
        entropy_out = results_at_pressure[0][2] + weight * (results_at_pressure[1][2] - results_at_pressure[0][2])
        volume_out = results_at_pressure[0][3] + weight * (results_at_pressure[1][3] - results_at_pressure[0][3])

    return dict(T=temperature_out, h=enthalpy_out, s=entropy_out, v=volume_out)


def sat_props(P):
    """Return saturated liquid and saturated vapor properties at pressure P (MPa)."""
    pressure_array = saturated_table["P"].to_numpy()
    if P < pressure_array[0] or P > pressure_array[-1]:
        raise ValueError(
            "Pressure " + str(P) + " MPa is outside the saturated steam table range ["
            + str(pressure_array[0]) + ", " + str(pressure_array[-1]) + "] MPa."
        )

    return dict(
        Tsat=float(np.interp(P, pressure_array, saturated_table["Tsat"])),
        vf=float(np.interp(P, pressure_array, saturated_table["vf"])),
        vg=float(np.interp(P, pressure_array, saturated_table["vg"])),
        hf=float(np.interp(P, pressure_array, saturated_table["hf"])),
        hg=float(np.interp(P, pressure_array, saturated_table["hg"])),
        sf=float(np.interp(P, pressure_array, saturated_table["sf"])),
        sg=float(np.interp(P, pressure_array, saturated_table["sg"])),
    )


def props_PT(P, T):
    """Return superheated steam properties at pressure P (MPa) and temperature T (degrees Celsius)."""
    saturation = sat_props(P)
    if T < saturation["Tsat"]:
        raise ValueError(
            "Temperature " + str(T) + " degrees Celsius is below the saturation temperature ("
            + "%.2f" % saturation["Tsat"] + " degrees Celsius) at " + str(P) + " MPa."
        )

    result = _interpolate_superheated_2d(P, T=T)
    return dict(h=result["h"], s=result["s"], v=result["v"])


def props_Ps(P, s):
    """Return steam properties at pressure P (MPa) and specific entropy s (kJ/kg.K)."""
    saturation = sat_props(P)
    if s < saturation["sf"]:
        raise ValueError(
            "Specific entropy " + str(s) + " kJ/(kg.K) is in the subcooled liquid region below "
            + "saturated liquid entropy " + "%.4f" % saturation["sf"] + " kJ/(kg.K)."
        )

    if s <= saturation["sg"]:
        quality = (s - saturation["sf"]) / (saturation["sg"] - saturation["sf"])
        enthalpy = saturation["hf"] + quality * (saturation["hg"] - saturation["hf"])
        volume = saturation["vf"] + quality * (saturation["vg"] - saturation["vf"])
        return dict(h=float(enthalpy), T=saturation["Tsat"], x=float(quality), v=float(volume))

    result = _interpolate_superheated_2d(P, s=s)
    return dict(h=result["h"], T=result["T"], x=1.0, v=result["v"])


def props_Ph(P, h):
    """Return steam properties at pressure P (MPa) and specific enthalpy h (kJ/kg)."""
    saturation = sat_props(P)
    if h < saturation["hf"]:
        raise ValueError(
            "Specific enthalpy " + str(h) + " kJ/kg is in the subcooled liquid region below "
            + "saturated liquid enthalpy " + "%.2f" % saturation["hf"] + " kJ/kg."
        )

    if h <= saturation["hg"]:
        quality = (h - saturation["hf"]) / (saturation["hg"] - saturation["hf"])
        entropy = saturation["sf"] + quality * (saturation["sg"] - saturation["sf"])
        volume = saturation["vf"] + quality * (saturation["vg"] - saturation["vf"])
        return dict(T=saturation["Tsat"], s=float(entropy), x=float(quality), v=float(volume))

    result = _interpolate_superheated_2d(P, h=h)
    return dict(T=result["T"], s=result["s"], x=1.0, v=result["v"])


if __name__ == "__main__":
    print("=== Interactive Steam Property Lookup ===")
    while True:
        print("\n1: Saturated properties by Pressure (sat_props)")
        print("2: Superheated properties by Pressure and Temperature (props_PT)")
        print("3: Two-phase or Superheated properties by Pressure and Entropy (props_Ps)")
        print("4: Two-phase or Superheated properties by Pressure and Enthalpy (props_Ph)")
        print("q: Quit")
        selected_mode = input("Select lookup mode: ").strip().lower()

        if selected_mode == "q":
            break
        try:
            pressure_input = float(input("Enter Pressure in MPa: "))
            if selected_mode == "1":
                print(sat_props(pressure_input))
            elif selected_mode == "2":
                temperature_input = float(input("Enter Temperature in degrees Celsius: "))
                print(props_PT(pressure_input, temperature_input))
            elif selected_mode == "3":
                entropy_input = float(input("Enter Specific Entropy in kJ/(kg.K): "))
                print(props_Ps(pressure_input, entropy_input))
            elif selected_mode == "4":
                enthalpy_input = float(input("Enter Specific Enthalpy in kJ/kg: "))
                print(props_Ph(pressure_input, enthalpy_input))
            else:
                print("Invalid mode selected.")
        except ValueError as error_message:
            print("Error:", error_message)
