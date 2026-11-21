# test_rankine.py
# pytest tests for src/rankine.py solve_cycle (Person A's function).
# Ideal-cycle targets are textbook values (verify with your own steam table):
# 15 MPa / 600 C / 1.2 MPa / 10 kPa -> y 0.227, x7 0.804, eta_th about 0.46.
# Units: P in MPa, T in C, h in kJ/kg. Efficiencies are fractions 0 to 1.

import copy

import pytest

from src.rankine import solve_cycle

REL = 0.01   # 1 percent tolerance (16.4)

BASE = {"T5": 600, "P5": 15.0, "P6": 1.2, "P7": 0.010,
        "eta_turb": 0.88, "eta_p1": 0.85, "eta_p2": 0.85, "mdot": 100}

IDEAL = dict(BASE, eta_turb=1.0, eta_p1=1.0, eta_p2=1.0)   # no losses


def _with(**changes):
    """Return a copy of the baseline params with some values changed."""
    p = copy.deepcopy(BASE)
    p.update(changes)
    return p


# ------------------------------------------------------------ ideal cycle
@pytest.fixture(scope="module")
def ideal():
    """Ideal cycle result (isentropic turbine and pumps)."""
    return solve_cycle(IDEAL)


def test_ideal_enthalpies(ideal):
    """States 5, 6 and 7 match the textbook enthalpies."""
    s = ideal["states"]
    assert s[5]["h"] == pytest.approx(3583.1, rel=REL)
    assert s[6]["h"] == pytest.approx(2860.2, rel=REL)
    assert s[7]["h"] == pytest.approx(2115.3, rel=REL)


def test_ideal_extraction_fraction(ideal):
    assert ideal["y"] == pytest.approx(0.227, rel=REL)


def test_ideal_exit_quality(ideal):
    assert ideal["x7"] == pytest.approx(0.804, rel=REL)
    assert ideal["states"][7]["x"] == pytest.approx(ideal["x7"])


def test_ideal_thermal_efficiency(ideal):
    """eta_th is a fraction near 0.46 (not 46 percent as a number)."""
    assert 0.0 < ideal["eta_th"] < 1.0
    assert ideal["eta_th"] == pytest.approx(0.46, abs=0.01)


# ------------------------------------------------------------ structure
def test_result_has_contract_keys():
    """Result dict follows section 16.5 and state keys are integers 1 to 7."""
    r = solve_cycle(BASE)
    for key in ("states", "y", "w_turb", "w_pump", "w_net", "q_in",
                "q_out", "eta_th", "x7", "power_MW"):
        assert key in r
    assert sorted(r["states"].keys()) == [1, 2, 3, 4, 5, 6, 7]
    for state in r["states"].values():
        for key in ("P", "T", "h", "s", "x"):
            assert key in state


def test_state_pressures_and_boiler_exit():
    """Pressures follow the diagram; state 5 is the boiler exit."""
    s = solve_cycle(BASE)["states"]
    assert s[1]["P"] == pytest.approx(0.010)
    assert s[2]["P"] == pytest.approx(1.2)
    assert s[3]["P"] == pytest.approx(1.2)
    assert s[4]["P"] == pytest.approx(15.0)
    assert s[5]["P"] == pytest.approx(15.0)
    assert s[5]["T"] == pytest.approx(600.0, rel=REL)
    assert s[6]["P"] == pytest.approx(1.2)
    assert s[7]["P"] == pytest.approx(0.010)


# ------------------------------------------------------------ balances
@pytest.mark.parametrize("params", [IDEAL, BASE])
def test_feedwater_heater_balance(params):
    """Open FWH energy balance: y*h6 + (1-y)*h2 = h3."""
    r = solve_cycle(params)
    h = {k: v["h"] for k, v in r["states"].items()}
    lhs = r["y"] * h[6] + (1 - r["y"]) * h[2]
    assert lhs == pytest.approx(h[3], rel=1e-4)


@pytest.mark.parametrize("params", [IDEAL, BASE])
def test_overall_energy_balance(params):
    """q_in - q_out = w_net, and w_net = w_turb - w_pump."""
    r = solve_cycle(params)
    assert r["w_net"] == pytest.approx(r["w_turb"] - r["w_pump"], rel=1e-6)
    assert r["q_in"] - r["q_out"] == pytest.approx(r["w_net"], rel=1e-3)


def test_power_matches_mass_flow():
    """power_MW = w_net * mdot / 1000."""
    r = solve_cycle(BASE)
    assert r["power_MW"] == pytest.approx(r["w_net"] * BASE["mdot"] / 1000.0)


def test_pump_work_units():
    """Pump work is a few to tens of kJ/kg; a kPa/MPa mix-up would be 1000x off."""
    r = solve_cycle(BASE)
    assert 5.0 < r["w_pump"] < 30.0
    assert r["w_pump"] < 0.05 * r["w_turb"]


def test_work_signs():
    """All work and heat values are positive magnitudes (16.3)."""
    r = solve_cycle(BASE)
    for key in ("w_turb", "w_pump", "w_net", "q_in", "q_out"):
        assert r[key] > 0


# ------------------------------------------------------------ trends
def test_lower_turbine_efficiency_lowers_eta_th():
    """Required check (16.10): lowering eta_turb lowers eta_th."""
    good = solve_cycle(_with(eta_turb=0.88))["eta_th"]
    bad = solve_cycle(_with(eta_turb=0.75))["eta_th"]
    assert bad < good


def test_real_cycle_is_worse_than_ideal(ideal):
    assert solve_cycle(BASE)["eta_th"] < ideal["eta_th"]


def test_lower_turbine_efficiency_gives_drier_exhaust():
    """Real expansion ends at higher h, so x7 goes up when eta_turb goes down."""
    assert solve_cycle(_with(eta_turb=0.75))["x7"] > solve_cycle(BASE)["x7"]


def test_higher_condenser_pressure_lowers_eta_th():
    """Fouled condenser (higher P7) hurts efficiency."""
    assert solve_cycle(_with(P7=0.015))["eta_th"] < solve_cycle(BASE)["eta_th"]


def test_lower_boiler_temperature_lowers_eta_th_and_x7():
    """Boiler fouling (lower T5) lowers eta_th and gives wetter exhaust."""
    hot, cool = solve_cycle(BASE), solve_cycle(_with(T5=560))
    assert cool["eta_th"] < hot["eta_th"]
    assert cool["x7"] < hot["x7"]


def test_worn_pumps_lower_eta_th():
    assert solve_cycle(_with(eta_p1=0.6, eta_p2=0.6))["eta_th"] < solve_cycle(BASE)["eta_th"]


# ------------------------------------------------------------ moisture loss (should have)
def test_moisture_factor_lowers_eta_th():
    """alpha > 0 lowers eta_th for wet steam (only once A adds moisture loss)."""
    plain = solve_cycle(BASE, alpha=0.0)["eta_th"]
    wet = solve_cycle(BASE, alpha=0.1)["eta_th"]
    assert wet < plain


# ------------------------------------------------------------ invalid input
def test_bad_extraction_pressure_raises():
    """P6 above P5 makes no sense: ValueError, not NaN."""
    with pytest.raises(ValueError):
        solve_cycle(_with(P6=20.0))


def test_bad_efficiency_raises():
    with pytest.raises(ValueError):
        solve_cycle(_with(eta_turb=1.5))
