"""
aeolus_rams_phase5.diagrams
=============================
Fault tree diagram rendering.

Primary renderer: graphviz (if available).
Fallback: text-based ASCII tree + matplotlib screenshot.

The graphviz renderer produces a directed graph with:
  - Rectangles for gates (labelled with name and Q value)
  - Ellipses for basic events (labelled with name and Q value)
  - Edges from gate to its inputs
  - AND gates annotated with 'AND' / '∧'
  - OR gates annotated with 'OR' / '∨'
  - Top event coloured red; gates coloured by type; basic events grey

Graphviz installation:
  pip install graphviz     # Python bindings
  apt install graphviz     # or: brew install graphviz (macOS)
  The Python package alone is not enough — the executable must be on PATH.
"""

from __future__ import annotations

from pathlib import Path

from .fault_tree import BasicEvent, Gate


def render_fault_tree(
    top_gate: Gate,
    output_path: Path,
    format: str = "png",
) -> Path:
    """Render the fault tree to a PNG (or SVG/PDF) file.

    Parameters
    ----------
    top_gate : Gate
        The fully-built overspeed fault tree.
    output_path : Path
        Output file path (extension determines format if format arg is auto).
    format : str
        Output format: 'png' (default), 'svg', 'pdf'.

    Returns
    -------
    Path
        Path to the rendered file.
    """
    try:
        return _render_graphviz(top_gate, output_path, format)
    except ImportError:
        return _render_matplotlib_fallback(top_gate, output_path)


def _render_graphviz(top_gate: Gate, output_path: Path, format: str) -> Path:
    """Render using the graphviz library."""
    import graphviz  # noqa: F401 — raises ImportError if not installed

    dot = graphviz.Digraph(
        name="AEOLUS-RAMS Phase 5 Fault Tree",
        graph_attr={
            "rankdir": "TB",
            "splines": "ortho",
            "fontname": "Helvetica",
            "fontsize": "11",
        },
        node_attr={"fontname": "Helvetica", "fontsize": "10"},
        edge_attr={"color": "#444444"},
    )

    # Gate colours
    GATE_COLORS = {"AND": "#FFCCCC", "OR": "#CCE5FF"}
    TOP_COLOR = "#FF4444"
    BE_COLOR = "#E8E8E8"

    node_counter = [0]

    def _add_node(node: Gate | BasicEvent, parent_id: str | None = None) -> str:
        node_counter[0] += 1
        node_id = f"n{node_counter[0]}"

        if isinstance(node, BasicEvent):
            label = (
                f"<<B>{node.name}</B><BR/>"
                f"Q={node.Q:.4e}<BR/>"
                f"<FONT POINT-SIZE='8'>{node.confidence}</FONT>>"
            )
            dot.node(node_id, label=label, shape="ellipse",
                     style="filled", fillcolor=BE_COLOR)
        else:
            gate_symbol = "∧" if node.gate_type == "AND" else "∨"
            is_top = parent_id is None
            fill = TOP_COLOR if is_top else GATE_COLORS[node.gate_type]
            label = (
                f"<<B>{gate_symbol} {node.gate_type}</B><BR/>"
                f"{node.name}<BR/>"
                f"<B>Q={node.Q:.4e}</B>>"
            )
            dot.node(node_id, label=label, shape="rectangle",
                     style="filled,rounded", fillcolor=fill)
            for child in node.inputs:
                child_id = _add_node(child, node_id)
                dot.edge(node_id, child_id)

        if parent_id is None:
            pass  # top gate has no parent edge
        return node_id

    _add_node(top_gate, None)

    # Render to file
    out_stem = str(output_path.with_suffix(""))
    dot.render(out_stem, format=format, cleanup=True)
    rendered_path = output_path.with_suffix(f".{format}")
    return rendered_path


def _render_matplotlib_fallback(top_gate: Gate, output_path: Path) -> Path:
    """Text-based fallback diagram rendered via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    lines = _tree_to_text_lines(top_gate)
    fig, ax = plt.subplots(figsize=(14, max(8, len(lines) * 0.35 + 2)))
    ax.axis("off")
    text = "\n".join(lines)
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontfamily="monospace",
        fontsize=9,
    )
    ax.set_title(
        f"AEOLUS-RAMS Phase 5 — Overspeed Fault Tree\n"
        f"Top Event Q = {top_gate.Q:.4e}  "
        f"(graphviz not available — install: pip install graphviz + apt install graphviz)",
        fontsize=10, pad=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _tree_to_text_lines(
    node: Gate | BasicEvent,
    prefix: str = "",
    is_last: bool = True,
) -> list[str]:
    """Recursively build text tree lines."""
    connector = "└── " if is_last else "├── "
    child_prefix = prefix + ("    " if is_last else "│   ")
    lines = []

    if isinstance(node, BasicEvent):
        lines.append(f"{prefix}{connector}[BE] {node.name}  Q={node.Q:.4e}")
    else:
        lines.append(
            f"{prefix}{connector}[{node.gate_type}] {node.name}  Q={node.Q:.4e}"
        )
        for i, child in enumerate(node.inputs):
            child_is_last = (i == len(node.inputs) - 1)
            lines.extend(_tree_to_text_lines(child, child_prefix, child_is_last))

    return lines


def text_fault_tree(top_gate: Gate) -> str:
    """Return the fault tree as a formatted text string (for report embedding)."""
    lines = [f"[AND] {top_gate.name}  Q={top_gate.Q:.4e}"]
    for i, child in enumerate(top_gate.inputs):
        is_last = (i == len(top_gate.inputs) - 1)
        lines.extend(_tree_to_text_lines(child, prefix="", is_last=is_last))
    return "\n".join(lines)