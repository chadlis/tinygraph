"""Tests for parallel fan-out via super-steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest

from tinygraph.reducers import add
from tinygraph.state import END, START, StateGraph

# ---------- Helpers ------------------------------------------------------------


def _make_branch(name: str) -> Callable[[S], dict[str, list[str]]]:
    return lambda s: {"log": [name]}


class S(TypedDict):
    log: Annotated[list[str], add]


# ---------- Happy paths --------------------------------------------------------


def test_diamond_fan_out_fan_in() -> None:
    """START → split → (a, b) → merge → END"""
    builder = StateGraph(S)
    builder.add_node("split", lambda s: {})
    builder.add_node("a", _make_branch("A"))
    builder.add_node("b", _make_branch("B"))
    builder.add_node("merge", _make_branch("merged"))

    builder.add_edge(START, "split")
    builder.add_edge("split", "a")
    builder.add_edge("split", "b")
    builder.add_edge("a", "merge")
    builder.add_edge("b", "merge")
    builder.add_edge("merge", END)

    result = builder.compile().invoke({"log": []})
    assert sorted(result["log"][:2]) == ["A", "B"]
    assert result["log"][2] == "merged"


def test_asymmetric_fan_out() -> None:
    """split → a → c → merge
     split → b --------→ merge
    merge waits for both branches."""
    builder = StateGraph(S)
    builder.add_node("split", lambda s: {})
    builder.add_node("a", _make_branch("A"))
    builder.add_node("b", _make_branch("B"))
    builder.add_node("c", _make_branch("C"))
    builder.add_node("merge", _make_branch("merged"))

    builder.add_edge(START, "split")
    builder.add_edge("split", "a")
    builder.add_edge("split", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "merge")
    builder.add_edge("c", "merge")
    builder.add_edge("merge", END)

    result = builder.compile().invoke({"log": []})
    assert "A" in result["log"]
    assert "B" in result["log"]
    assert "C" in result["log"]
    assert result["log"][-1] == "merged"


def test_three_way_fan_out() -> None:
    """split fans out to 3 branches, all merge back."""
    builder = StateGraph(S)
    builder.add_node("split", lambda s: {})
    builder.add_node("a", _make_branch("A"))
    builder.add_node("b", _make_branch("B"))
    builder.add_node("c", _make_branch("C"))
    builder.add_node("merge", _make_branch("merged"))

    builder.add_edge(START, "split")
    builder.add_edge("split", "a")
    builder.add_edge("split", "b")
    builder.add_edge("split", "c")
    builder.add_edge("a", "merge")
    builder.add_edge("b", "merge")
    builder.add_edge("c", "merge")
    builder.add_edge("merge", END)

    result = builder.compile().invoke({"log": []})
    assert sorted(result["log"][:3]) == ["A", "B", "C"]
    assert result["log"][3] == "merged"


def test_parallel_nodes_read_same_snapshot() -> None:
    """Parallel nodes see the same state, not each other's updates."""

    class CountState(TypedDict):
        count: int

    builder = StateGraph(CountState)
    builder.add_node("split", lambda s: {})
    builder.add_node("a", lambda s: {"count": s["count"] + 1})
    builder.add_node("b", lambda s: {"count": s["count"] + 1})
    builder.add_node("merge", lambda s: {})

    builder.add_edge(START, "split")
    builder.add_edge("split", "a")
    builder.add_edge("split", "b")
    builder.add_edge("a", "merge")
    builder.add_edge("b", "merge")
    builder.add_edge("merge", END)

    result = builder.compile().invoke({"count": 0})
    # Both read count=0, both write count=1. No reducer → last wins = 1, not 2.
    assert result["count"] == 1


def test_recursion_limit_counts_super_steps() -> None:
    """The limit counts ticks, not individual node executions."""
    builder = StateGraph(S)
    builder.add_node("split", lambda s: {})
    builder.add_node("a", _make_branch("A"))
    builder.add_node("b", _make_branch("B"))
    builder.add_node("merge", _make_branch("merged"))

    builder.add_edge(START, "split")
    builder.add_edge("split", "a")
    builder.add_edge("split", "b")
    builder.add_edge("a", "merge")
    builder.add_edge("b", "merge")
    builder.add_edge("merge", END)

    graph = builder.compile()
    # Diamond takes 4 ticks: split, (a,b), merge, END → limit=4 should work
    result = graph.invoke({"log": []}, recursion_limit=4)
    assert result["log"][-1] == "merged"

    # limit=2 → only split and (a,b) execute, merge tick fails
    with pytest.raises(RecursionError):
        graph.invoke({"log": []}, recursion_limit=2)


def test_fan_out_coexists_with_conditional_edge() -> None:
    """Fan-out on one node, conditional routing on another, same graph."""

    class RS(TypedDict):
        log: Annotated[list[str], add]
        route: str

    builder = StateGraph(RS)
    builder.add_node("split", lambda s: {})
    builder.add_node("a", _make_branch("A"))
    builder.add_node("b", _make_branch("B"))
    builder.add_node("router", lambda s: {"log": ["routed"]})
    builder.add_node("x", _make_branch("X"))
    builder.add_node("y", _make_branch("Y"))

    builder.add_edge(START, "split")
    builder.add_edge("split", "a")
    builder.add_edge("split", "b")
    builder.add_edge("a", "router")
    builder.add_edge("b", "router")
    builder.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"go_x": "x", "go_y": "y"},
    )
    builder.add_edge("x", END)
    builder.add_edge("y", END)

    result = builder.compile().invoke({"log": [], "route": "go_x"})
    assert "A" in result["log"]
    assert "B" in result["log"]
    assert "routed" in result["log"]
    assert "X" in result["log"]
    assert "Y" not in result["log"]
