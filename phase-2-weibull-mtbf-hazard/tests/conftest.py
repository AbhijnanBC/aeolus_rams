"""
Synthetic fixtures for Phase 2.

Builds TWO consistent things for a fake farm "A":
1. A raw, CARE-shaped `data_root` — one windowed SCADA episode file per
   logged event (matching CARE's real structure, where
   `n_turbine_datasets == n_logged_events` per farm — see Section 2.1.1),
   each file carrying a real `asset_id`.
2. A matching `phase1_dir` containing `tagged_events.csv` (the events,
   pre-tagged with a `component_primary`) and `fmeca_table.csv` — built
   by calling the REAL, installed Phase 1 package's
   `aeolus_rams.fmeca.build_fmeca_table` on those same events, so the
   fixture is schema-consistent with genuine Phase 1 output by
   construction, not hand-typed to match.

The component -> per-asset incident schedule is deliberately chosen to
exercise three real scenarios in one fixture:
  - Pitch System: 11 incidents split 9+2 across two assets -> 9 usable
    (uncensored) TBF intervals -> stays Tier A.
  - Gearbox: 8 incidents split 5+3 across two assets -> only 6 usable
    TBF intervals -> Section 2.3's "usable TBF < incident count" trap,
    triggering pipeline.py's tier re-validation (A -> B downgrade).
  - Hydraulic System: 6 incidents, single asset -> 5 usable intervals ->
    stays Tier B (right at the boundary).
  - Every other taxonomy component gets 0-4 incidents -> Tier C.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeolus_rams.taxonomy import COMPONENT_NAMES
from aeolus_rams.fmeca import build_fmeca_table

FARM = "A"

# component -> list of incident counts, one entry per (fresh) asset
SCHEDULE: dict[str, list[int]] = {
    "Pitch System": [9, 2],
    "Gearbox": [5, 3],
    "Hydraulic System": [6],
    "Generator": [3],
    "Transformer": [2],
    "Mechanical Brake": [1],
    "Converter": [4],
    "Yaw System": [2],
    "SCADA/Communication": [1],
    "Electrical Safety System": [3],
    "Cooling System": [0],
    "Grounding/Lightning Protection": [0],
    "Main/Rotor Bearing": [2],
}
assert set(SCHEDULE.keys()) == set(COMPONENT_NAMES), (
    set(COMPONENT_NAMES) - set(SCHEDULE.keys()),
    set(SCHEDULE.keys()) - set(COMPONENT_NAMES),
)

EVENT_SPACING_DAYS = 45
EPISODE_PADDING = pd.Timedelta(hours=6)
EPISODE_DURATION = pd.Timedelta(hours=6)


def _write_episode_file(datasets_dir, filename, asset_id, window_start, window_end, seed):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(window_start, window_end, freq="10min")
    n = len(timestamps)
    status = rng.choice([0, 0, 0, 0, 1, 3, 4], size=n)
    df = pd.DataFrame({
        "time_stamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "asset_id": asset_id,
        "id": np.arange(n),
        "train_test": ["test"] * n,
        "status_type_id": status,
    })
    df.to_csv(datasets_dir / filename, sep=";", index=False)


def _build_phase2_fixture(care_root, phase1_dir, seed: int = 0):
    farm_dir = care_root / f"Wind Farm {FARM}" / f"Wind Farm {FARM}"
    datasets_dir = farm_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)

    event_rows = []
    event_id = 0
    file_idx = 0
    next_asset_id = 0
    base_time = pd.Timestamp("2022-01-01")
    rng = np.random.default_rng(seed)

    for component, per_asset_counts in SCHEDULE.items():
        for n_incidents in per_asset_counts:
            asset_id = next_asset_id
            next_asset_id += 1
            asset_start = base_time + pd.Timedelta(days=3650 * asset_id)

            # Jittered spacing (+/- 30%) so synthetic TBF sequences aren't
            # perfectly identical — a degenerate all-equal sample makes
            # scipy's Weibull MLE numerically unstable (huge beta) and
            # isn't representative of real failure data anyway.
            cumulative_days = 0.0
            for i in range(n_incidents):
                if i > 0:
                    jitter = rng.uniform(0.7, 1.3)
                    cumulative_days += EVENT_SPACING_DAYS * jitter
                event_start = asset_start + pd.Timedelta(days=cumulative_days)
                event_end = event_start + EPISODE_DURATION
                window_start = event_start - EPISODE_PADDING
                window_end = event_end + EPISODE_PADDING

                filename = f"{file_idx}.csv"
                _write_episode_file(
                    datasets_dir, filename, asset_id, window_start, window_end,
                    seed=seed * 10_000 + file_idx,
                )
                file_idx += 1

                event_rows.append({
                    "farm": FARM,
                    "event_id": event_id,
                    "event_label": "anomaly",
                    "event_start": event_start,
                    "event_start_id": 1000 + event_id,
                    "event_end": event_end,
                    "event_end_id": 1000 + event_id + 36,
                    "event_description": f"synthetic {component} incident {i}",
                    "component_primary": component,
                    "component_secondary": None,
                    "tag_confidence": "curated",
                    "tag_notes": "synthetic fixture",
                    "additional_systems_mentioned": "",
                    "needs_manual_review": False,
                    "_true_asset_id": asset_id,  # test-only ground truth
                })
                event_id += 1

    # A couple of normal-labeled events (no component, no incident) —
    # must be correctly ignored everywhere downstream.
    for i in range(2):
        asset_id = next_asset_id
        next_asset_id += 1
        event_start = base_time + pd.Timedelta(days=900 + 10 * i)
        event_end = event_start + EPISODE_DURATION
        filename = f"{file_idx}.csv"
        _write_episode_file(
            datasets_dir, filename, asset_id,
            event_start - EPISODE_PADDING, event_end + EPISODE_PADDING,
            seed=seed * 10_000 + file_idx,
        )
        file_idx += 1
        event_rows.append({
            "farm": FARM, "event_id": event_id, "event_label": "normal",
            "event_start": event_start, "event_start_id": 2000 + event_id,
            "event_end": event_end, "event_end_id": 2000 + event_id + 36,
            "event_description": None, "component_primary": "Unclassified / No Description",
            "component_secondary": None, "tag_confidence": "no_description",
            "tag_notes": "", "additional_systems_mentioned": "",
            "needs_manual_review": False, "_true_asset_id": asset_id,
        })
        event_id += 1

    events_df = pd.DataFrame(event_rows)

    # aeolus_rams.data_loader.discover_farms anchors farm discovery on
    # event_info.csv's presence (rglob), so the raw CARE tree needs one
    # even though Phase 2 itself reads tagged_events.csv from phase1_dir,
    # not this file directly.
    event_info_cols = [
        "event_id", "event_label", "event_start", "event_start_id",
        "event_end", "event_end_id", "event_description",
    ]
    events_df[event_info_cols].to_csv(farm_dir / "event_info.csv", sep=";", index=False)
    pd.DataFrame({
        "sensor_name": ["sensor_0"], "statistics_type": ["average"],
        "description": ["placeholder"], "unit": ["-"],
        "is_angle": [False], "is_counter": [False],
    }).to_csv(farm_dir / "feature_description.csv", sep=";", index=False)

    tagged_events = events_df.drop(columns=["_true_asset_id"])
    tagged_events.to_csv(phase1_dir / "tagged_events.csv", index=False)

    # Build a genuinely real, schema-correct fmeca_table.csv by calling
    # the actual installed Phase 1 package on these same tagged events.
    fmeca_table = build_fmeca_table(tagged_events)
    fmeca_table.to_csv(phase1_dir / "fmeca_table.csv", index=False)

    return events_df  # includes _true_asset_id for test assertions


@pytest.fixture
def phase2_project(tmp_path):
    """Returns (data_root, phase1_dir, ground_truth_events_df)."""
    data_root = tmp_path / "care"
    phase1_dir = tmp_path / "phase-1-system-fmeca"
    ground_truth = _build_phase2_fixture(data_root, phase1_dir, seed=0)
    return data_root, phase1_dir, ground_truth
