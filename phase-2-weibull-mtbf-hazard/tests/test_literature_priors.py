from __future__ import annotations

import math

import pytest

from aeolus_rams.taxonomy import COMPONENT_NAMES
from aeolus_rams_phase2.literature_priors import (
    LITERATURE_PRIORS, get_prior, tier_c_r_of_t,
    CITED_DIRECT, DERIVED, DERIVED_ASSUMED_RATING, NOT_YET_SOURCED,
)


def test_all_taxonomy_components_have_an_entry():
    assert set(LITERATURE_PRIORS.keys()) == set(COMPONENT_NAMES)


def test_get_prior_known_component():
    prior = get_prior("Gearbox")
    assert prior.mtbf_days is not None
    assert prior.confidence == CITED_DIRECT


def test_get_prior_unknown_component_raises():
    with pytest.raises(KeyError):
        get_prior("Not A Real Component")


def test_every_prior_has_a_nonempty_source():
    for prior in LITERATURE_PRIORS.values():
        assert prior.source.strip() != ""
        assert prior.derivation_note.strip() != ""


def test_every_confidence_value_is_one_of_the_known_labels():
    known = {CITED_DIRECT, DERIVED, DERIVED_ASSUMED_RATING, NOT_YET_SOURCED}
    for prior in LITERATURE_PRIORS.values():
        assert prior.confidence in known


def test_not_yet_sourced_priors_have_no_mtbf():
    for prior in LITERATURE_PRIORS.values():
        if prior.confidence == NOT_YET_SOURCED:
            assert prior.mtbf_days is None
        else:
            assert prior.mtbf_days is not None


def test_is_usable_matches_mtbf_presence():
    for prior in LITERATURE_PRIORS.values():
        assert prior.is_usable == (prior.mtbf_days is not None)


def test_main_bearing_derivation_matches_documented_calculation():
    # lambda = -ln(1 - 0.30) / 20 years; MTBF = 1/lambda in days.
    prior = get_prior("Main/Rotor Bearing")
    lam = -math.log(1 - 0.30) / 20
    expected_days = (1 / lam) * 365.25
    assert prior.mtbf_days == pytest.approx(expected_days, rel=0.01)


def test_gearbox_and_generator_match_carroll_2016_rates():
    gearbox = get_prior("Gearbox")
    generator = get_prior("Generator")
    assert gearbox.mtbf_days == pytest.approx(365.25 / 0.154, rel=0.01)
    assert generator.mtbf_days == pytest.approx(365.25 / 0.095, rel=0.01)


def test_tier_c_r_of_t_exponential_formula():
    prior = get_prior("Gearbox")
    r = tier_c_r_of_t(prior, prior.mtbf_days)
    assert r == pytest.approx(math.exp(-1), rel=1e-9)  # R(MTBF) = e^-1 under exponential


def test_tier_c_r_of_t_returns_none_for_unsourced_prior():
    prior = get_prior("Transformer")  # not_yet_sourced in this file
    assert tier_c_r_of_t(prior, 100) is None


def test_tier_c_r_of_t_decreasing_in_time():
    prior = get_prior("Gearbox")
    r_early = tier_c_r_of_t(prior, 10)
    r_late = tier_c_r_of_t(prior, 1000)
    assert r_early > r_late
