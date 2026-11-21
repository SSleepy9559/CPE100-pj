# test_steam_props.py
# pytest tests for src/steam_props.py (Person A's functions).
# Reference values: textbook steam tables (Cengel style). VERIFY them with your own
# steam table or CoolProp before relying on them. Tolerance about 1 percent (16.4).
# Units: P in MPa, T in C, h in kJ/kg, s in kJ/(kg.K), v in m3/kg.

import pytest

from src.steam_props import props_Ph, props_Ps, props_PT, sat_props

REL = 0.01   # 1 percent tolerance


# ------------------------------------------------------------ sat_props
# (P [MPa], Tsat, vf, hf, hg, sf, sg) from textbook saturation table
SAT_CASES = [
    (0.010, 45.81, 0.001010, 191.81, 2583.9, 0.6492, 8.1488),
    (1.2, 187.96, 0.001138, 798.33, 2784.6, 2.2159, 6.5233),
]


@pytest.mark.parametrize("P, Tsat, vf, hf, hg, sf, sg", SAT_CASES)
def test_sat_props_textbook(P, Tsat, vf, hf, hg, sf, sg):
    """Saturation properties match the textbook table."""
    r = sat_props(P)
    assert r["Tsat"] == pytest.approx(Tsat, rel=REL)
    assert r["vf"] == pytest.approx(vf, rel=REL)
    assert r["hf"] == pytest.approx(hf, rel=REL)
    assert r["hg"] == pytest.approx(hg, rel=REL)
    assert r["sf"] == pytest.approx(sf, rel=REL)
    assert r["sg"] == pytest.approx(sg, rel=REL)


def test_sat_props_has_all_keys():
    """sat_props returns the keys promised in the contract (16.7)."""
    r = sat_props(0.5)
    for key in ("Tsat", "vf", "vg", "hf", "hg", "sf", "sg"):
        assert key in r


def test_sat_props_trends():
    """Tsat and hf rise with pressure; hg > hf and sg > sf always."""
    low, high = sat_props(0.1), sat_props(2.0)
    assert high["Tsat"] > low["Tsat"]
    assert high["hf"] > low["hf"]
    for r in (low, high):
        assert r["hg"] > r["hf"]
        assert r["sg"] > r["sf"]


# ------------------------------------------------------------ props_PT
def test_props_PT_boiler_exit():
    """15 MPa, 600 C superheated steam (state 5)."""
    r = props_PT(15.0, 600.0)
    assert r["h"] == pytest.approx(3583.1, rel=REL)
    assert r["s"] == pytest.approx(6.6796, rel=REL)


def test_props_PT_enthalpy_rises_with_temperature():
    """At fixed pressure, hotter superheated steam has higher h and s."""
    cold, hot = props_PT(1.2, 300.0), props_PT(1.2, 500.0)
    assert hot["h"] > cold["h"]
    assert hot["s"] > cold["s"]


# ------------------------------------------------------------ props_Ps
def test_props_Ps_extraction_state_is_superheated():
    """Isentropic expansion to 1.2 MPa from state 5 (state 6): h about 2860, x = 1."""
    s5 = props_PT(15.0, 600.0)["s"]
    r = props_Ps(1.2, s5)
    assert r["h"] == pytest.approx(2860.2, rel=REL)
    assert r["x"] == pytest.approx(1.0)          # superheated, never None or NaN
    assert r["T"] > sat_props(1.2)["Tsat"]


def test_props_Ps_turbine_exit_is_wet():
    """Isentropic expansion to 10 kPa (state 7): x about 0.804, h about 2115."""
    s5 = props_PT(15.0, 600.0)["s"]
    r = props_Ps(0.010, s5)
    assert r["x"] == pytest.approx(0.804, rel=REL)
    assert r["h"] == pytest.approx(2115.3, rel=REL)
    assert r["T"] == pytest.approx(45.81, rel=REL)   # wet steam is at Tsat


# ------------------------------------------------------------ props_Ph
def test_props_Ph_quality_from_enthalpy():
    """In the two-phase region x = (h - hf) / (hg - hf)."""
    sat = sat_props(0.010)
    h = 0.5 * (sat["hf"] + sat["hg"])
    r = props_Ph(0.010, h)
    assert r["x"] == pytest.approx(0.5, rel=REL)


def test_props_Ph_superheated_x_is_one():
    """Superheated steam must give x = 1.0, not None or NaN."""
    r = props_Ph(1.2, 2860.2)
    assert r["x"] == pytest.approx(1.0)
    assert r["s"] == pytest.approx(6.6796, rel=REL)


# ------------------------------------------------------------ consistency
@pytest.mark.parametrize("P, T", [(1.2, 400.0), (5.0, 500.0), (15.0, 600.0)])
def test_round_trip_PT_Ps_Ph(P, T):
    """Looking up with (P,T), then (P,s) and (P,h), gives the same state."""
    base = props_PT(P, T)
    from_s = props_Ps(P, base["s"])
    from_h = props_Ph(P, base["h"])
    assert from_s["h"] == pytest.approx(base["h"], rel=REL)
    assert from_s["T"] == pytest.approx(T, rel=REL)
    assert from_h["s"] == pytest.approx(base["s"], rel=REL)
    assert from_h["T"] == pytest.approx(T, rel=REL)


# ------------------------------------------------------------ invalid input
@pytest.mark.parametrize("P", [-1.0, 0.0, 1000.0])
def test_sat_props_bad_pressure_raises(P):
    """Pressure outside the table range raises ValueError (16.7), never NaN."""
    with pytest.raises(ValueError):
        sat_props(P)


def test_props_PT_bad_input_raises():
    """Out-of-range pressure or temperature raises ValueError."""
    with pytest.raises(ValueError):
        props_PT(-1.0, 600.0)
    with pytest.raises(ValueError):
        props_PT(15.0, 5000.0)
