"""
aeolus_rams.data_loader
=========================
Robust discovery and I/O for a CARE-to-Compare download, regardless of
exact unzip nesting depth (Section 1.8: "CARE's own distribution nests
each farm inside a doubled folder name ... that's inherent to how the
archive unzips, not something you need to reorganize").

All functions here are pure I/O + light parsing — no scoring, no tagging,
no FMECA logic lives in this module.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger("aeolus_rams.data_loader")

# Disable PyArrow string inference to avoid hanging on wide CSVs
pd.options.mode.string_storage = "python"
try:
    pd.options.io.parquet.engine = "pyarrow"
except AttributeError:
    pass  # Older pandas versions don't have this option


class CAREDataError(RuntimeError):
    """Raised when the CARE directory tree doesn't look like CARE."""


@dataclass(frozen=True)
class FarmPaths:
    """Resolved filesystem layout for a single wind farm."""

    farm_id: str
    root: Path

    @property
    def event_info_csv(self) -> Path:
        return self.root / "event_info.csv"

    @property
    def feature_description_csv(self) -> Path:
        return self.root / "feature_description.csv"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    def dataset_files(self) -> list[Path]:
        """All per-turbine SCADA dataset CSVs for this farm, sorted for
        determinism (natural/numeric sort where filenames are numeric,
        e.g. "0.csv", "1.csv", ..., "10.csv")."""
        files = list(self.datasets_dir.glob("*.csv"))

        def _sort_key(p: Path):
            stem = p.stem
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        return sorted(files, key=_sort_key)


# ---------------------------------------------------------------------------
# Farm discovery
# ---------------------------------------------------------------------------

def _infer_farm_id(event_info_path: Path) -> str:
    """Recover a short farm identifier (e.g. "A") from a discovered
    event_info.csv path.

    Strategy:
    1. Look for the canonical "Wind Farm <X>" pattern anywhere in the path
       (robust to arbitrary nesting depth / doubled folders).
    2. Fall back to the last character of the immediate parent folder name
       — this matches the heuristic used in the Phase 1 spec's reference
       snippet (Section 1.8) and the original recon notebook, so farm IDs
       stay consistent with any earlier exploratory work.
    """
    match = re.search(config.FARM_NAME_PATTERN, str(event_info_path))
    if match:
        return match.group(1).upper()
    return event_info_path.parent.name[-1].upper()


def discover_farms(data_root: str | Path) -> dict[str, FarmPaths]:
    """Auto-discover every wind farm under `data_root`, regardless of
    nesting depth, by locating every `event_info.csv` file.

    Raises
    ------
    CAREDataError
        If `data_root` doesn't exist, or no `event_info.csv` is found
        anywhere beneath it, or two different directories resolve to the
        same farm id (ambiguous / duplicated download).
    """
    root = Path(data_root)
    if not root.exists():
        raise CAREDataError(
            f"data_root does not exist: {root.resolve()}\n"
            "Point --data-root at the folder that CONTAINS your unzipped "
            "CARE download (e.g. data/raw/care), not a specific farm "
            "subfolder."
        )

    event_files = sorted(root.rglob("event_info.csv"))
    if not event_files:
        raise CAREDataError(
            f"No event_info.csv found anywhere under {root.resolve()}. "
            "Confirm the CARE archive was unzipped under this path."
        )

    farm_paths: dict[str, FarmPaths] = {}
    for f in event_files:
        farm_id = _infer_farm_id(f)
        if farm_id in farm_paths and farm_paths[farm_id].root != f.parent:
            raise CAREDataError(
                f"Ambiguous farm id '{farm_id}': found event_info.csv under "
                f"both {farm_paths[farm_id].root} and {f.parent}. Resolve "
                "the duplicate download before continuing."
            )
        farm_paths[farm_id] = FarmPaths(farm_id=farm_id, root=f.parent)

    logger.info("Discovered %d farm(s): %s", len(farm_paths), sorted(farm_paths))
    return dict(sorted(farm_paths.items()))


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def load_event_table(farm: FarmPaths) -> pd.DataFrame:
    """Load and lightly clean a farm's event_info.csv.

    - Parses `event_start` / `event_end` as datetimes.
    - Leaves `event_description` untouched (raw free text — normalisation
      is the tagger's responsibility, not the loader's).
    """
    path = farm.event_info_csv
    if not path.exists():
        raise CAREDataError(f"Missing event_info.csv for farm {farm.farm_id}: {path}")

    df = pd.read_csv(path, sep=config.CARE_CSV_SEPARATOR)
    missing_cols = set(config.EVENT_INFO_COLUMNS) - set(df.columns)
    if missing_cols:
        raise CAREDataError(
            f"Farm {farm.farm_id} event_info.csv is missing expected "
            f"columns: {sorted(missing_cols)}"
        )

    for col in ("event_start", "event_end"):
        df[col] = pd.to_datetime(df[col])

    df.insert(0, "farm", farm.farm_id)
    return df


def load_feature_table(farm: FarmPaths) -> pd.DataFrame:
    """Load a farm's feature_description.csv (sensor metadata)."""
    path = farm.feature_description_csv
    if not path.exists():
        raise CAREDataError(
            f"Missing feature_description.csv for farm {farm.farm_id}: {path}"
        )
    df = pd.read_csv(path, sep=config.CARE_CSV_SEPARATOR)
    df.insert(0, "farm", farm.farm_id)
    return df


def load_dataset_file(
    path: Path,
    columns: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a single per-turbine SCADA dataset CSV using pure Python.

    Parameters
    ----------
    columns : restrict to specific columns (e.g. just the status column)
        for speed on wide (up to ~957-column) farm-C files.
    nrows : cap the number of rows read (useful for fast structural checks
        on files that can run to 60k+ rows).
    """
    # Read complete CSV using pure Python - robust and handles all platforms.
    # On PowerShell/Linux this should work without signal interrupts.
    rows = []
    header = None
    selected_indices = None
    
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=config.CARE_CSV_SEPARATOR)
            
            for i, row in enumerate(reader):
                if i == 0:  # Header row
                    header = row
                    if columns:
                        # Find indices of requested columns
                        try:
                            selected_indices = [header.index(c) for c in columns if c in header]
                        except ValueError as e:
                            logger.warning("Column not found in %s: %s", path.name, e)
                            selected_indices = None
                    continue
                
                if nrows and len(rows) >= nrows:
                    break  # Stop if we've read enough rows
                
                # Extract selected columns if specified, else keep all
                if selected_indices is not None:
                    row_data = [row[idx] if idx < len(row) else "" for idx in selected_indices]
                else:
                    row_data = row
                
                rows.append(row_data)
        
        # Build dataframe
        if selected_indices is not None and header is not None:
            col_names = [header[idx] for idx in selected_indices]
        else:
            col_names = header if header else []
        
        if not rows:
            logger.warning("Empty file: %s", path)
            return pd.DataFrame(columns=col_names)
        
        df = pd.DataFrame(rows, columns=col_names)
        
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        raise
    
    # Convert specific columns to proper types for downstream processing
    if "time_stamp" in df.columns:
        try:
            df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        except Exception as e:
            logger.warning("Failed to parse time_stamp in %s: %s", path, e)
    
    # Convert numeric columns to native types for scientific calculations
    for col in df.columns:
        if col != "time_stamp":
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass  # Keep as-is if conversion fails
    
    return df


def load_all_event_tables(
    farm_paths: dict[str, FarmPaths],
) -> dict[str, pd.DataFrame]:
    """Convenience: load every farm's event_info.csv into a dict."""
    return {farm_id: load_event_table(fp) for farm_id, fp in farm_paths.items()}


def inventory_summary(farm_paths: dict[str, FarmPaths]) -> pd.DataFrame:
    """Section 1.1.3 — basic inventory sanity check, as a DataFrame instead
    of print statements so it's testable and report-friendly."""
    rows = []
    for farm_id, fp in farm_paths.items():
        n_datasets = len(fp.dataset_files())
        events = load_event_table(fp)
        n_anomaly = int((events["event_label"] == "anomaly").sum())
        n_normal = int((events["event_label"] == "normal").sum())
        rows.append({
            "farm": farm_id,
            "n_turbine_datasets": n_datasets,
            "n_logged_events": len(events),
            "n_anomaly_events": n_anomaly,
            "n_normal_events": n_normal,
            "has_feature_description": fp.feature_description_csv.exists(),
            "counts_reconcile": n_anomaly + n_normal == len(events),
        })
    return pd.DataFrame(rows)
