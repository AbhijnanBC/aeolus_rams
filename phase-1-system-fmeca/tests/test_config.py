from __future__ import annotations

from aeolus_rams import config


def test_status_legend_has_six_codes():
    assert set(config.STATUS_LEGEND.keys()) == {0, 1, 2, 3, 4, 5}


def test_fault_status_id_is_marked_not_normal():
    assert config.STATUS_LEGEND[config.FAULT_STATUS_ID].considered_normal is False


def test_normal_and_abnormal_status_ids_partition_the_legend():
    normal = config.normal_status_ids()
    abnormal = config.abnormal_status_ids()
    assert normal | abnormal == set(config.STATUS_LEGEND.keys())
    assert normal & abnormal == set()


def test_canonical_schema_status_column_is_status_type_id():
    # Regression guard: an earlier recon pass incorrectly assumed the
    # status column was named "status_type" — it is "status_type_id".
    assert config.CANONICAL_SCHEMA["status"] == "status_type_id"


def test_canonical_schema_covers_expected_fields():
    for field in ("timestamp", "asset_id", "status", "event_label"):
        assert field in config.CANONICAL_SCHEMA


def test_scope_boundary_excludes_balance_of_plant():
    for item in ("array cables", "offshore substation", "export cable"):
        assert item in config.PHASE1_SCOPE.out_of_scope
    assert "gearbox" not in config.PHASE1_SCOPE.out_of_scope
