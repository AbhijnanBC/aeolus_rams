"""
aeolus_rams_phase6
==================
Phase 6 of the AEOLUS-RAMS pipeline: Event Tree Analysis (ETA).

Reads Phase 5 fault tree outputs (gate_Q_table.csv, ccf_sensitivity.csv,
importance_table.csv) and produces:
  - Consequence frequency table across λ_IE sensitivity range
  - IEC 61400-1 risk matrix with ETA branches and BoP cable (Phase 4)
  - ALARP quantification for three barrier improvement options
  - Bow-tie diagram integrating FTA (Phase 5) and ETA (Phase 6)
  - Full Markdown report

All Q values are read directly from Phase 5 outputs — no hard-coding.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aeolus-rams-phase6")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = [
    "config",
    "event_tree",
    "risk_matrix",
    "bowtie",
    "alarp",
    "reporting",
    "pipeline",
]