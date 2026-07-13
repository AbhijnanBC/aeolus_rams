from __future__ import annotations

import pytest

from aeolus_rams import taxonomy as tax


def test_taxonomy_has_thirteen_components():
    assert len(tax.COMPONENT_TAXONOMY) == 13


def test_component_names_are_unique():
    names = tax.COMPONENT_NAMES
    assert len(names) == len(set(names))


def test_priority_order_covers_every_component_exactly_once():
    assert sorted(tax.PRIORITY_ORDER) == sorted(tax.COMPONENT_NAMES)
    assert len(tax.PRIORITY_ORDER) == len(set(tax.PRIORITY_ORDER))


def test_keyword_rules_cover_every_component():
    covered = {name for name, _ in tax.KEYWORD_RULES}
    assert covered == set(tax.COMPONENT_NAMES)


def test_keyword_rules_have_nonempty_keyword_lists():
    for component, keywords in tax.KEYWORD_RULES:
        assert len(keywords) > 0, component


def test_component_by_name_found():
    c = tax.component_by_name("Gearbox")
    assert c.subsystem_group == "Drivetrain"


def test_component_by_name_missing_raises_helpful_error():
    with pytest.raises(KeyError):
        tax.component_by_name("Not A Real Component")
