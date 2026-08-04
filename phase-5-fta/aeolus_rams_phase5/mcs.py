"""
aeolus_rams_phase5.mcs
========================
Minimal Cut Set (MCS) extraction and ranking.

Algorithm: recursive Boolean tree traversal using the
'product-of-sums' (MOCUS-equivalent) method:
  - AND gate: Cartesian product of child MCS lists
  - OR gate:  union of child MCS lists

For this specific tree (AND top with three OR sub-gates, each with
2–3 basic events), the algorithm produces exactly 3×2×2 = 12 MCS,
all of order 3. This was verified by hand before any code was written.

All MCS are of order 3 — this means no single failure and no pair of
failures can cause overspeed. This is a strong IEC 61400-1 design
property and should be stated explicitly in the Phase 5 report.

References:
    Vesely, W.E. et al. (1981) "NUREG-0492: Fault Tree Handbook."
    U.S. Nuclear Regulatory Commission.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from .fault_tree import BasicEvent, Gate


TreeNode = Union[BasicEvent, Gate]


def extract_minimal_cut_sets(top_gate: Gate) -> list[frozenset[str]]:
    """Extract all minimal cut sets by recursive tree traversal.

    Returns a list of frozensets, where each frozenset contains the
    names of BasicEvents whose simultaneous failure is necessary and
    sufficient to cause the top event.

    'Minimal' means: removing any event from the set would break the
    ability to cause the top event (i.e., no set is a superset of another).

    Parameters
    ----------
    top_gate : Gate
        The top-level gate of the fault tree.

    Returns
    -------
    list[frozenset[str]]
        Sorted by MCS Q descending (highest-probability MCS first).
    """
    all_be = top_gate.collect_basic_events()

    def _mcs_from_node(node: TreeNode) -> list[frozenset[str]]:
        if isinstance(node, BasicEvent):
            return [frozenset([node.name])]

        if node.gate_type == "AND":
            # Cartesian product: every MCS must contain one element
            # from each child's MCS list
            result: list[frozenset[str]] = [frozenset()]
            for child in node.inputs:
                child_mcs = _mcs_from_node(child)
                result = [a | b for a in result for b in child_mcs]
        else:  # OR gate
            # Union: any child's MCS can cause this gate to be 'failed'
            result = []
            for child in node.inputs:
                result.extend(_mcs_from_node(child))

        return result

    raw_mcs = _mcs_from_node(top_gate)

    # Minimality check: remove supersets
    minimal: list[frozenset[str]] = []
    for candidate in raw_mcs:
        dominated = any(
            other < candidate   # other is a proper subset
            for other in raw_mcs
            if other is not candidate
        )
        if not dominated:
            minimal.append(candidate)

    # Sort by MCS Q descending
    def _mcs_q(mcs: frozenset[str]) -> float:
        q = 1.0
        for name in mcs:
            q *= all_be[name].Q
        return q

    return sorted(minimal, key=_mcs_q, reverse=True)


def mcs_table(
    top_gate: Gate,
    t_label: str = "365d",
) -> pd.DataFrame:
    """Build a DataFrame of all MCS with Q values and importance metrics.

    Columns
    -------
    rank : int
        Rank by Q (1 = highest probability MCS).
    mcs_id : str
        Label: 'MCS-1', 'MCS-2', ...
    events : str
        Comma-separated basic event names.
    order : int
        Number of events in the MCS (all should be 3 for this tree).
    Q_mcs : float
        Product of basic event Q values (probability of this specific
        failure scenario).
    FV : float
        Fussell-Vesely contribution: Q_mcs / sum(all Q_mcs).
        Uses rare-event approximation (valid when Q_mcs << 1).
    contains_G1a, G1b, G1c, G2a, G2b, G3a, G3b : bool
        Which basic events appear in this MCS (for filtering).
    t : str
        Mission time label.
    """
    mcs_list = extract_minimal_cut_sets(top_gate)
    all_be = top_gate.collect_basic_events()

    be_names = list(all_be.keys())

    def _mcs_q(mcs: frozenset[str]) -> float:
        q = 1.0
        for name in mcs:
            q *= all_be[name].Q
        return q

    q_values = [_mcs_q(m) for m in mcs_list]
    q_total = sum(q_values)   # for FV denominator (rare-event approx)

    rows = []
    for i, (mcs, q) in enumerate(zip(mcs_list, q_values), start=1):
        row = {
            "rank": i,
            "mcs_id": f"MCS-{i}",
            "events": " ∩ ".join(sorted(mcs)),
            "order": len(mcs),
            "Q_mcs": q,
            "FV": q / q_total if q_total > 0 else 0.0,
            "t": t_label,
        }
        # Per-event membership columns (short labels)
        short_map = {be.name: be.name.split(":")[0].strip() for be in all_be.values()}
        for name in be_names:
            col = f"has_{name.split(':')[0].strip().replace(' ', '_')}"
            row[col] = name in mcs
        rows.append(row)

    return pd.DataFrame(rows)