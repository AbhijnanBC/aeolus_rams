from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aeolus_rams_phase3.component_rt import (
    load_all_components, load_mtbf_table, inject_option_a_placeholders,
    build_component_rt, lambda_per_component, lambda_system, mtbf_system,
    reliability_table,
)
from aeolus_rams_phase3 import config
from tests.conftest import EXPECTED_PLACEHOLDER_MTBF


# ---------------------------------------------------------------------------
# load_mtbf_table + inject_option_a_placeholders
# ---------------------------------------------------------------------------

def test_load_mtbf_table_reads_all_13(mtbf_table_path):
    df = load_mtbf_table(mtbf_table_path)
    assert len(df) == 13


def test_load_mtbf_table_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mtbf_table(tmp_path / "does_not_exist.csv")


def test_load_mtbf_table_wrong_schema_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_mtbf_table(path)


def test_inject_placeholders_fills_nan(mtbf_table_path):
    df = load_mtbf_table(mtbf_table_path)
    df_injected = inject_option_a_placeholders(df)
    assert df_injected["mtbf_days"].isna().sum() == 0


def test_inject_placeholders_tags_confidence(mtbf_table_path):
    df = load_mtbf_table(mtbf_table_path)
    df_injected = inject_option_a_placeholders(df)
    ph_rows = df_injected[df_injected["is_placeholder"]]
    assert (ph_rows["confidence"] == "assumed_placeholder").all()


def test_inject_placeholders_exact_values(mtbf_table_path):
    df = load_mtbf_table(mtbf_table_path)
    df_injected = inject_option_a_placeholders(df)
    for comp, expected_mtbf in EXPECTED_PLACEHOLDER_MTBF.items():
        row = df_injected[df_injected["component"] == comp].iloc[0]
        assert row["mtbf_days"] == pytest.approx(expected_mtbf, rel=1e-9)


def test_inject_placeholders_unknown_nan_raises(mtbf_table_path_with_bad_nan):
    df = load_mtbf_table(mtbf_table_path_with_bad_nan)
    with pytest.raises(ValueError, match="has no entry in"):
        inject_option_a_placeholders(df)


def test_non_placeholder_rows_untouched(mtbf_table_path):
    df = load_mtbf_table(mtbf_table_path)
    df_injected = inject_option_a_placeholders(df)
    gearbox = df_injected[df_injected["component"] == "Gearbox"].iloc[0]
    assert gearbox["mtbf_days"] == pytest.approx(28033.147886957817, rel=1e-6)
    assert not gearbox["is_placeholder"]
    assert gearbox["confidence"] == "posterior_informed"


# ---------------------------------------------------------------------------
# load_all_components — the master entry point
# ---------------------------------------------------------------------------

def test_load_all_components_returns_13(mtbf_table_path):
    comps = load_all_components(mtbf_table_path)
    assert len(comps) == 13


def test_all_components_have_positive_lambda(components):
    for name, c in components.items():
        assert c.lambda_per_day > 0, f"{name}: lambda ≤ 0"
        assert math.isfinite(c.lambda_per_day), f"{name}: lambda is not finite"


def test_no_nan_in_any_component_mtbf(components):
    for name, c in components.items():
        assert not math.isnan(c.mtbf_days), f"{name} still has NaN MTBF"


def test_six_components_are_placeholders(components):
    n_ph = sum(1 for c in components.values() if c.is_placeholder)
    assert n_ph == 6


def test_seven_components_are_not_placeholders(components):
    n_real = sum(1 for c in components.values() if not c.is_placeholder)
    assert n_real == 7


# ---------------------------------------------------------------------------
# ComponentRT.R(t) — core reliability function
# ---------------------------------------------------------------------------

def test_R_at_zero_is_one(components):
    for name, c in components.items():
        assert c.R(0.0) == pytest.approx(1.0, abs=1e-12), f"{name}: R(0)≠1"


def test_R_monotonically_decreasing(components):
    t_vals = np.linspace(1, 10000, 200)
    for name, c in components.items():
        r = c.R(t_vals)
        assert np.all(np.diff(r) <= 0), f"{name}: R(t) is not monotonically decreasing"


def test_R_approaches_zero_at_large_t(components):
    for name, c in components.items():
        r = c.R(1e7)
        assert r < 0.001, f"{name}: R(1e7 days) = {r} is not near 0"


def test_pitch_R_at_1yr_matches_spec(components):
    # Spec Section 3.2: Pitch System, λ=1/1936.38, R(365.25)≈0.8288
    pitch = components["Pitch System"]
    expected = math.exp(-365.25 / 1936.3776678104537)
    assert pitch.R(365.25) == pytest.approx(expected, rel=1e-9)
    assert pitch.R(365.25) == pytest.approx(0.8288, abs=0.001)


def test_hydraulic_R_at_1yr_matches_spec(components):
    # Spec Section 3.2: Hydraulic System, λ=0.000542, R(365.25)≈0.8204
    hydro = components["Hydraulic System"]
    expected = math.exp(-365.25 / 1844.7037037037037)
    assert hydro.R(365.25) == pytest.approx(expected, rel=1e-9)
    assert hydro.R(365.25) == pytest.approx(0.8204, abs=0.001)


def test_R_scalar_and_array_consistent(components):
    pitch = components["Pitch System"]
    t_scalar = 365.25
    t_array = np.array([365.25])
    assert pitch.R(t_scalar) == pytest.approx(float(pitch.R(t_array)[0]), rel=1e-12)


def test_aic_preferred_exponential_flag(components):
    pitch = components["Pitch System"]
    assert pitch.uses_aic_preferred_exponential is True
    # All other Tier A or non-Tier-A components should not carry this flag
    for name, c in components.items():
        if name != "Pitch System":
            assert c.uses_aic_preferred_exponential is False


# ---------------------------------------------------------------------------
# Aggregate functions
# ---------------------------------------------------------------------------

def test_lambda_system_is_sum_of_all_lambdas(components):
    expected = sum(1.0 / c.mtbf_days for c in components.values())
    assert lambda_system(components) == pytest.approx(expected, rel=1e-12)


def test_mtbf_system_is_reciprocal_of_lambda_system(components):
    lam = lambda_system(components)
    assert mtbf_system(components) == pytest.approx(1.0 / lam, rel=1e-12)


def test_reliability_table_has_all_components(components):
    tbl = reliability_table(components)
    assert set(tbl["component"]) == set(components.keys())


def test_reliability_table_no_nan(components):
    tbl = reliability_table(components)
    r_cols = [c for c in tbl.columns if c.startswith("R_")]
    assert not tbl[r_cols].isna().any().any()


def test_lambda_per_component_keys_match(components):
    lams = lambda_per_component(components)
    assert set(lams.keys()) == set(components.keys())
    for name, lam in lams.items():
        assert lam == pytest.approx(components[name].lambda_per_day, rel=1e-12)
