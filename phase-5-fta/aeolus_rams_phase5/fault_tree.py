"""
aeolus_rams_phase5.fault_tree
==============================
Defines the data model for the AEOLUS-RAMS overspeed fault tree and
implements the gate Q-value computation.

Architecture: tree nodes are immutable dataclasses. The tree is a pure
data structure — no mutation after build. This means:
  - Q values at any node are computed on demand (property).
  - Perturbing one basic event's Q (for Birnbaum importance) requires
    building a modified tree, not monkey-patching a live object.
  - Tests can verify gate formulas against hand-calculated values exactly.

Gate Q algebra:
  OR gate:  Q_OR  = 1 − ∏ᵢ (1 − Qᵢ)          [exact]
  AND gate: Q_AND = ∏ᵢ Qᵢ                      [assumes independence]

The independence assumption at the AND gate is examined by ccf.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union
import numpy as np
import pandas as pd



# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BasicEvent:
    """A leaf node — a primary failure event with a known Q(t).

    Attributes
    ----------
    name : str
        Unique identifier. Used as the key in MCS frozensets and
        in the importance table.
    Q : float
        Probability of being in the failed state at mission time t.
        For overspeed FTA, this is used as P(failure to respond on demand)
        — a standard engineering proxy when demand-specific test data
        is unavailable.
    source : str
        Provenance tag: 'phase3_rt_table', 'phase3_rt_table×fraction_estimate',
        or 'ccf_pathway'.
    confidence : str
        Inherited from Phase 3: 'fitted_tier_a', 'fitted_tier_b',
        'posterior_informed', 'assumed_placeholder'.
    parent_component : str
        The Phase 3 component name this basic event derives from.
    fraction_estimate : float
        The sub-cause fraction applied to the parent component Q.
        0.0 for basic events that use the full parent Q.
    """
    name: str
    Q: float
    source: str = ""
    confidence: str = ""
    parent_component: str = ""
    fraction_estimate: float = 0.0

    def with_Q(self, new_Q: float) -> "BasicEvent":
        """Return a copy of this event with a different Q value.
        Used for Birnbaum importance computation (perturb-and-evaluate).
        """
        return BasicEvent(
            name=self.name, Q=new_Q, source=self.source,
            confidence=self.confidence, parent_component=self.parent_component,
            fraction_estimate=self.fraction_estimate,
        )


@dataclass(frozen=True)
class Gate:
    """An intermediate node combining sub-events with AND or OR logic.

    Attributes
    ----------
    name : str
        Descriptive label for the gate (appears in diagram and report).
    gate_type : 'AND' | 'OR'
        Boolean combination rule.
    inputs : list of BasicEvent or Gate
        Child nodes. The list is immutable (tuple internally for hashability).
    """
    name: str
    gate_type: Literal["AND", "OR"]
    inputs: tuple[Union["BasicEvent", "Gate"], ...]

    @property
    def Q(self) -> float:
        """Compute gate Q from child Q values using the exact formula.

        OR gate:  Q = 1 − ∏ᵢ (1 − Qᵢ)    [exact inclusion-exclusion]
        AND gate: Q = ∏ᵢ Qᵢ               [independence assumed]

        Note: the OR gate formula never uses the rare-event approximation
        (Q_OR ≈ ΣQᵢ). That approximation overestimates risk and is only
        used in the Fussell-Vesely calculation where it is the standard
        definition.
        """
        if self.gate_type == "OR":
            result = 1.0
            for inp in self.inputs:
                result *= (1.0 - inp.Q)
            return 1.0 - result
        else:  # AND — independence assumed (examined by ccf.py)
            result = 1.0
            for inp in self.inputs:
                result *= inp.Q
            return result

    def with_basic_event_Q(
        self,
        event_name: str,
        new_Q: float,
    ) -> "Gate":
        """Return a copy of this tree with one basic event's Q replaced.

        Used for Birnbaum importance (evaluate Q_top with Qᵢ=0 and Qᵢ=1).
        Rebuilds the minimum number of Gate objects (only those on the path
        from root to the modified event).
        """
        new_inputs = []
        for inp in self.inputs:
            if isinstance(inp, BasicEvent):
                if inp.name == event_name:
                    new_inputs.append(inp.with_Q(new_Q))
                else:
                    new_inputs.append(inp)
            else:
                new_inputs.append(inp.with_basic_event_Q(event_name, new_Q))
        return Gate(name=self.name, gate_type=self.gate_type, inputs=tuple(new_inputs))

    def collect_basic_events(self) -> dict[str, BasicEvent]:
        """Return a flat dict of all basic events in this subtree."""
        result: dict[str, BasicEvent] = {}
        for inp in self.inputs:
            if isinstance(inp, BasicEvent):
                result[inp.name] = inp
            else:
                result.update(inp.collect_basic_events())
        return result

    def collect_gates(self) -> dict[str, "Gate"]:
        """Return a flat dict of all gates (including self) in this subtree."""
        result: dict[str, "Gate"] = {self.name: self}
        for inp in self.inputs:
            if isinstance(inp, Gate):
                result.update(inp.collect_gates())
        return result


# ---------------------------------------------------------------------------
# Sub-cause Q normalisation
# ---------------------------------------------------------------------------

def _allocate_subcause_Q(
    Q_total: float,
    fractions: list[float],
) -> list[float]:
    """Allocate sub-cause Q values so their OR-gate result equals Q_total.

    The OR gate is nonlinear: simply splitting Q_total by fractions gives
    an OR result slightly below Q_total (because individual Qᵢ are positive,
    so joint terms reduce the OR sum). This function scales the sub-event
    Q values until the OR gate matches Q_total exactly.

    Parameters
    ----------
    Q_total : float
        Target OR gate output (= parent component Q from Phase 3).
    fractions : list[float]
        Must sum to 1.0. Relative shares of each sub-cause.

    Returns
    -------
    list[float]
        Q values for each sub-cause, in the same order as fractions.
        OR-gate of the returned list equals Q_total to within 1e-12.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9, "Fractions must sum to 1.0"
    assert 0.0 <= Q_total <= 1.0, f"Q_total={Q_total} out of [0,1]"

    # Initial allocation: Qᵢ = Q_total × fᵢ
    Q_subs = [Q_total * f for f in fractions]

    # Compute OR gate result and rescale if needed
    for _ in range(100):  # Newton-ish iteration — usually converges in 2-3 steps
        Q_or = 1.0 - np.prod([1.0 - q for q in Q_subs])
        if abs(Q_or - Q_total) < 1e-12:
            break
        if Q_or == 0.0:
            break
        scale = Q_total / Q_or
        Q_subs = [q * scale for q in Q_subs]

    return Q_subs


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

def build_overspeed_fault_tree(
    Q_1yr: dict[str, float],
    sub_cause_fractions: dict | None = None,
) -> Gate:
    """Build the turbine overspeed fault tree from Phase 3 Q(1yr) values.

    Parameters
    ----------
    Q_1yr : dict[str, float]
        Maps component name → Q at 1 year. Read from Phase 3 component_rt_table.csv
        via config.load_Q_values(). Accepted keys: at minimum the five components
        used in the tree (Pitch System, Hydraulic System, Mechanical Brake,
        SCADA/Communication, Electrical Safety System).
    sub_cause_fractions : dict, optional
        Override default sub-cause fractions from config. Used in sensitivity sweep.
        Must have keys 'G1', 'G2', 'G3', each a list of fractions summing to 1.0.

    Returns
    -------
    Gate
        The top-event gate (AND gate). Access Q_top via top_gate.Q.

    Tree Structure
    --------------
    TE (AND) ← G1 (OR) + G2 (OR) + G3 (OR)
    G1 ← G1a (encoder/control) + G1b (battery/electrical) + G1c (actuator mechanical)
    G2 ← G2a (hydraulic supply) + G2b (brake hardware)
    G3 ← G3a (fieldbus/communication) + G3b (safety relay fault)

    Sub-cause Q normalisation (enforced by _allocate_subcause_Q):
        G1 OR-gate result = Q_pitch (Phase 3 exact value)
        G2 OR-gate result = computed (no fixed parent target — uses Hydraulic + Brake Q)
        G3 OR-gate result = computed (SCADA + Electrical Safety Q)

    Common-cause note:
        G2a uses Q_hydraulic (not Q_brake) as its source component — because the
        hydraulic system physically feeds both pitch actuators (G1) and brake caliper
        (G2). This is the CCF pathway examined in ccf.py. See Phase 1 FMECA event:
        "Rotorbrake and Hydraulic problemes — Hydraulic pump A disabled."
    """
    from . import config as _cfg

    # Apply fraction overrides if provided
    g1_fracs = (
        sub_cause_fractions.get("G1", list(_cfg.G1_FRACTIONS.values()))
        if sub_cause_fractions
        else list(_cfg.G1_FRACTIONS.values())
    )
    g2_fracs = (
        sub_cause_fractions.get("G2", list(_cfg.G2_FRACTIONS.values()))
        if sub_cause_fractions
        else list(_cfg.G2_FRACTIONS.values())
    )
    g3_fracs = (
        sub_cause_fractions.get("G3", list(_cfg.G3_FRACTIONS.values()))
        if sub_cause_fractions
        else list(_cfg.G3_FRACTIONS.values())
    )

    Q_pitch = Q_1yr["Pitch System"]
    Q_hyd   = Q_1yr["Hydraulic System"]
    Q_brake = Q_1yr["Mechanical Brake"]
    Q_scada = Q_1yr["SCADA/Communication"]
    Q_esys  = Q_1yr["Electrical Safety System"]

    # ── G1: Pitch fails to feather (OR, 3 sub-causes) ─────────────────────
    # Sub-event Q values normalised so G1.Q = Q_pitch exactly.
    g1_fracs_sum_check = sum(g1_fracs)
    g1_fracs_normalised = [f / g1_fracs_sum_check for f in g1_fracs]
    g1a_Q, g1b_Q, g1c_Q = _allocate_subcause_Q(Q_pitch, g1_fracs_normalised)

    g1a = BasicEvent(
        name="G1a: Control/Encoder fault",
        Q=g1a_Q,
        source="phase3_rt_table[Pitch System] × fraction_estimate",
        confidence="fitted_tier_a",
        parent_component="Pitch System",
        fraction_estimate=g1_fracs_normalised[0],
    )
    g1b = BasicEvent(
        name="G1b: Battery/Electrical supply fault",
        Q=g1b_Q,
        source="phase3_rt_table[Pitch System] × fraction_estimate",
        confidence="fitted_tier_a",
        parent_component="Pitch System",
        fraction_estimate=g1_fracs_normalised[1],
    )
    g1c = BasicEvent(
        name="G1c: Actuator mechanical failure",
        Q=g1c_Q,
        source="phase3_rt_table[Pitch System] × fraction_estimate",
        confidence="fitted_tier_a",
        parent_component="Pitch System",
        fraction_estimate=g1_fracs_normalised[2],
    )
    G1 = Gate("G1: Pitch System fails to feather", "OR", (g1a, g1b, g1c))

    # ── G2: Brake fails to engage (OR, 2 sub-causes) ──────────────────────
    # G2a source is Hydraulic System Q — shared infrastructure with G1 (CCF pathway).
    # G2b source is Mechanical Brake Q — direct brake hardware failure.
    # No parent Q normalisation here: G2's Q is determined by the OR of hydraulic
    # supply loss and brake hardware fault (two genuinely different physical components).
    g2_fracs_normalised = [f / sum(g2_fracs) for f in g2_fracs]
    g2a_Q = Q_hyd   * g2_fracs_normalised[0]   # 60% of hydraulic Q
    g2b_Q = Q_brake * g2_fracs_normalised[1]   # 40% of brake Q

    g2a = BasicEvent(
        name="G2a: Hydraulic supply loss",
        Q=g2a_Q,
        source="phase3_rt_table[Hydraulic System] × fraction_estimate",
        confidence="fitted_tier_b",
        parent_component="Hydraulic System",
        fraction_estimate=g2_fracs_normalised[0],
    )
    g2b = BasicEvent(
        name="G2b: Brake hardware fault",
        Q=g2b_Q,
        source="phase3_rt_table[Mechanical Brake] × fraction_estimate",
        confidence="assumed_placeholder",
        parent_component="Mechanical Brake",
        fraction_estimate=g2_fracs_normalised[1],
    )
    G2 = Gate("G2: Mechanical brake fails to engage", "OR", (g2a, g2b))

    # ── G3: SCADA overspeed trip fails (OR, 2 sub-causes) ─────────────────
    # G3a: communication/fieldbus loss (BK1120 module, NC300 fault — CARE events)
    # G3b: safety relay / electrical safety fault (safety chain relay, RCD fault)
    g3_fracs_normalised = [f / sum(g3_fracs) for f in g3_fracs]
    g3a_Q = Q_scada * g3_fracs_normalised[0]
    g3b_Q = Q_esys  * g3_fracs_normalised[1]

    g3a = BasicEvent(
        name="G3a: Communication/fieldbus loss",
        Q=g3a_Q,
        source="phase3_rt_table[SCADA/Communication] × fraction_estimate",
        confidence="posterior_informed",
        parent_component="SCADA/Communication",
        fraction_estimate=g3_fracs_normalised[0],
    )
    g3b = BasicEvent(
        name="G3b: Safety relay/electrical fault",
        Q=g3b_Q,
        source="phase3_rt_table[Electrical Safety System] × fraction_estimate",
        confidence="assumed_placeholder",
        parent_component="Electrical Safety System",
        fraction_estimate=g3_fracs_normalised[1],
    )
    G3 = Gate("G3: SCADA overspeed trip fails to activate", "OR", (g3a, g3b))

    # ── Top Event: AND gate — all three layers must fail ──────────────────
    TE = Gate(
        "TE: Turbine Overspeed → Catastrophic Structural Failure",
        "AND",
        (G1, G2, G3),
    )
    return TE


def gate_Q_table(top_gate: Gate, t_label: str = "365d") -> "pd.DataFrame":
    """Build a table of Q values for every gate and basic event in the tree.

    Returns a DataFrame with columns: node, node_type, gate_type_or_source,
    Q, parent_component, confidence, t_label.
    """
    import pandas as pd

    rows: list[dict] = []

    def _visit(node: Gate | BasicEvent, depth: int = 0) -> None:
        indent = "  " * depth
        if isinstance(node, BasicEvent):
            rows.append({
                "node": f"{indent}{node.name}",
                "node_type": "BasicEvent",
                "gate_type": "—",
                "Q": node.Q,
                "confidence": node.confidence,
                "parent_component": node.parent_component,
                "fraction_estimate": node.fraction_estimate,
                "t": t_label,
            })
        else:
            rows.append({
                "node": f"{indent}{node.name}",
                "node_type": "Gate",
                "gate_type": node.gate_type,
                "Q": node.Q,
                "confidence": "computed",
                "parent_component": "—",
                "fraction_estimate": 1.0,
                "t": t_label,
            })
            for inp in node.inputs:
                _visit(inp, depth + 1)

    _visit(top_gate)
    return pd.DataFrame(rows)