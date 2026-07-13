"""
Synthetic CARE-shaped fixtures.

We do NOT ship the real CARE dataset (it's several GB and licensed
separately — see the paper's Zenodo DOI). Instead we build a tiny,
schema-faithful synthetic tree on disk for every test: same doubled
folder nesting, same `;` separator, same column names, and a subset of
the REAL event_description text pulled from this project's own recon
pass, so component_tagger's curated lookup table is exercised against
genuine strings, not test-only stand-ins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# A small, real subset of event_description text per farm (verbatim,
# taken from this project's own CARE download) — used so tests exercise
# the actual curated lookup table, not synthetic stand-ins.
REAL_DESCRIPTIONS: dict[str, list[str]] = {
    "A": [
        "Transformer failure",
        "Hydraulic group",
        "Gearbox failure",
        "Generator bearing failure",
    ],
    "B": [
        "high temperature",
        "Rotor Bearing 2 - Damage",
        "Turbine is stopped due to a main bearing damage",
    ],
    "C": [
        "Converter Failure from 17.11 12:30 - 18.11. 13:57, Fuse Filter Supply",
        "23020 : Axis 3 not ready-to-operate",
        "PENDING19_PREV_YAW_Grease pump defective",
        "Failure due to Rotorbrake and Hydraulic problemes - Hydraulic pump A "
        "disabeld, 2h later turbine back in production - Data shows anomaly in "
        "temp_hydraulic_oil_tank_1_average until 15.01.2023",
    ],
}

SENSOR_COLUMNS = [
    "sensor_0_avg", "sensor_0_max", "sensor_0_min", "sensor_0_std",
    "sensor_1_avg", "power_2_avg", "wind_speed_3_avg",
]


def _make_event_info(farm: str, n_normal: int = 3, repeat_first: int = 2) -> pd.DataFrame:
    descs = REAL_DESCRIPTIONS[farm]
    # Repeat the first description to exercise value_counts()-based
    # occurrence counting (Section 1.1.2) rather than just unique text.
    desc_sequence = [descs[0]] * repeat_first + descs[1:]

    rows = []
    event_id = 0
    start = pd.Timestamp("2022-01-01")

    for i, desc in enumerate(desc_sequence):
        s = start + pd.Timedelta(days=10 * i)
        e = s + pd.Timedelta(hours=6)
        rows.append({
            "event_id": event_id,
            "event_label": "anomaly",
            "event_start": s.strftime("%Y-%m-%d %H:%M:%S"),
            "event_start_id": 1000 + event_id,
            "event_end": e.strftime("%Y-%m-%d %H:%M:%S"),
            "event_end_id": 1000 + event_id + 36,
            "event_description": desc,
        })
        event_id += 1

    for i in range(n_normal):
        s = start + pd.Timedelta(days=300 + 10 * i)
        e = s + pd.Timedelta(hours=6)
        rows.append({
            "event_id": event_id,
            "event_label": "normal",
            "event_start": s.strftime("%Y-%m-%d %H:%M:%S"),
            "event_start_id": 2000 + event_id,
            "event_end": e.strftime("%Y-%m-%d %H:%M:%S"),
            "event_end_id": 2000 + event_id + 36,
            "event_description": None,
        })
        event_id += 1

    return pd.DataFrame(rows)


def _make_feature_description() -> pd.DataFrame:
    return pd.DataFrame({
        "sensor_name": ["sensor_0", "sensor_1", "power_2", "wind_speed_3"],
        "statistics_type": [
            "maximum,minimum,average,std_dev", "average", "average", "average",
        ],
        "description": [
            "Ambient temperature", "Wind absolute direction", "Active power",
            "Windspeed",
        ],
        "unit": ["degC", "deg", "kW", "m/s"],
        "is_angle": [False, True, False, False],
        "is_counter": [False, False, False, False],
    })


def _make_turbine_dataset(asset_id: int, n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + asset_id)
    timestamps = pd.date_range("2022-01-01", periods=n_rows, freq="10min")

    # Mostly-normal status, with a modest fraction of fault/derate/service
    # codes so status-legend checks have something real to count.
    status = rng.choice(
        [0, 0, 0, 0, 0, 1, 3, 4, 5],
        size=n_rows,
    )

    data = {
        "time_stamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "asset_id": asset_id,
        "id": np.arange(n_rows),
        "train_test": ["train"] * n_rows,
        "status_type_id": status,
    }
    for col in SENSOR_COLUMNS:
        data[col] = rng.normal(loc=20.0, scale=5.0, size=n_rows).round(3)

    return pd.DataFrame(data)


def _build_farm(root, farm: str, n_turbines: int = 2, n_rows: int = 300) -> None:
    # CARE's real archive nests each farm inside a doubled folder name,
    # e.g. "Wind Farm A/Wind Farm A/" — reproduced here deliberately so
    # data_loader's discovery logic is tested against the real layout.
    farm_dir = root / f"Wind Farm {farm}" / f"Wind Farm {farm}"
    datasets_dir = farm_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    _make_event_info(farm).to_csv(farm_dir / "event_info.csv", sep=";", index=False)
    _make_feature_description().to_csv(
        farm_dir / "feature_description.csv", sep=";", index=False
    )
    for turbine in range(n_turbines):
        df = _make_turbine_dataset(asset_id=turbine, n_rows=n_rows)
        df.to_csv(datasets_dir / f"{turbine}.csv", sep=";", index=False)


@pytest.fixture
def care_root(tmp_path):
    """A complete, on-disk, schema-faithful synthetic CARE download with
    3 farms (A, B, C), matching the real doubled-folder nesting."""
    root = tmp_path / "care"
    for farm in ("A", "B", "C"):
        _build_farm(root, farm)
    return root


@pytest.fixture
def care_root_single_farm(tmp_path):
    """A minimal single-farm tree, for tests that don't need all three."""
    root = tmp_path / "care_single"
    _build_farm(root, "A", n_turbines=1, n_rows=150)
    return root
