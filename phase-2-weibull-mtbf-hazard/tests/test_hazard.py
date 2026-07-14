from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase2.hazard import (
    weibull_hazard, exponential_hazard, exponential_hazard_curve,
    render_component_hazard, render_illustrative_system_bathtub,
)
from aeolus_rams_phase2.distributions import fit_weibull, fit_exponential_only
from aeolus_rams_phase2.literature_priors import LITERATURE_PRIORS
from scipy import stats


def test_weibull_hazard_increasing_for_beta_gt_1():
    t = np.linspace(1, 500, 50)
    h = weibull_hazard(t, beta=2.5, eta=100.0)
    assert np.all(np.diff(h) > 0)


def test_weibull_hazard_decreasing_for_beta_lt_1():
    t = np.linspace(1, 500, 50)
    h = weibull_hazard(t, beta=0.5, eta=100.0)
    assert np.all(np.diff(h) < 0)


def test_weibull_hazard_constant_for_beta_eq_1():
    t = np.linspace(1, 500, 50)
    h = weibull_hazard(t, beta=1.0, eta=100.0)
    assert np.allclose(h, h[0])


def test_exponential_hazard_is_scalar_constant():
    assert exponential_hazard(0.01) == 0.01


def test_exponential_hazard_curve_is_flat_array():
    t = np.linspace(1, 100, 20)
    h = exponential_hazard_curve(t, 0.02)
    assert np.allclose(h, 0.02)
    assert h.shape == t.shape


def test_render_component_hazard_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    fit_a = fit_weibull(stats.weibull_min.rvs(2.0, scale=100.0, size=10, random_state=rng))
    fit_b = fit_exponential_only(stats.expon.rvs(scale=80.0, size=6, random_state=rng))

    out = tmp_path / "hazard.png"
    path = render_component_hazard(
        {"Pitch System": fit_a}, {"Hydraulic System": fit_b},
        ["Gearbox", "Transformer"], out,
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_render_component_hazard_handles_no_fits(tmp_path):
    out = tmp_path / "hazard_empty.png"
    path = render_component_hazard({}, {}, ["Gearbox"], out)
    assert path.exists()


def test_render_illustrative_system_bathtub_writes_file(tmp_path):
    rng = np.random.default_rng(1)
    fit_a = fit_weibull(stats.weibull_min.rvs(2.0, scale=100.0, size=10, random_state=rng))
    tier_c = {k: v for k, v in LITERATURE_PRIORS.items() if v.mtbf_days is not None}

    out = tmp_path / "bathtub.png"
    path = render_illustrative_system_bathtub({"Pitch System": fit_a}, {}, tier_c, out)
    assert path.exists()
    assert path.stat().st_size > 0


def test_render_illustrative_system_bathtub_skips_unsourced_priors(tmp_path):
    # Priors with mtbf_days=None must not crash the composite plot.
    unsourced = {k: v for k, v in LITERATURE_PRIORS.items() if v.mtbf_days is None}
    assert unsourced, "expected at least one not_yet_sourced prior in the fixture data"
    out = tmp_path / "bathtub_unsourced.png"
    path = render_illustrative_system_bathtub({}, {}, unsourced, out, t_max_days=1000)
    assert path.exists()
