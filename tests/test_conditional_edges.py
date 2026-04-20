"""Tests for conditional edges in StateGraph."""

from __future__ import annotations

from typing import TypedDict

import pytest

from tinygraph.graph import CompiledGraph
from tinygraph.state import END, START, StateGraph

# ---------- Fixtures / helpers -------------------------------------------------


class SignState(TypedDict):
    value: int
    label: str


def _route_sign(state: SignState) -> str:
    if state["value"] > 0:
        return "pos"
    if state["value"] < 0:
        return "neg"
    return "zero"


def _build_sign_graph() -> CompiledGraph[SignState]:
    builder = StateGraph(SignState)
    builder.add_node("classify", lambda s: {})
    builder.add_node("pos_branch", lambda s: {"label": "positive"})
    builder.add_node("neg_branch", lambda s: {"label": "negative"})
    builder.add_node("zero_branch", lambda s: {"label": "zero"})
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        _route_sign,
        {"pos": "pos_branch", "neg": "neg_branch", "zero": "zero_branch"},
    )
    builder.add_edge("pos_branch", END)
    builder.add_edge("neg_branch", END)
    builder.add_edge("zero_branch", END)
    return builder.compile()


# ---------- Happy paths --------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(7, "positive"), (-4, "negative"), (0, "zero")],
)
def test_conditional_routing_three_branches(value: int, expected: str) -> None:
    graph = _build_sign_graph()
    result = graph.invoke({"value": value, "label": ""})
    assert result["label"] == expected


def test_conditional_edge_can_target_end_directly() -> None:
    class S(TypedDict):
        x: int

    builder = StateGraph(S)
    builder.add_node("a", lambda s: {"x": s["x"] + 1})
    builder.add_node("b", lambda s: {"x": s["x"] + 10})
    builder.add_edge(START, "a")
    builder.add_conditional_edges(
        "a",
        lambda s: "stop" if s["x"] == 1 else "continue",
        {"stop": END, "continue": "b"},
    )
    builder.add_edge("b", END)
    graph = builder.compile()

    # x=0 -> a (+1 = 1) -> "stop" -> END
    assert graph.invoke({"x": 0})["x"] == 1
    # x=5 -> a (+1 = 6) -> "continue" -> b (+10 = 16) -> END
    assert graph.invoke({"x": 5})["x"] == 16


def test_successors_includes_conditional_targets() -> None:
    class S(TypedDict):
        x: int

    builder = StateGraph(S)
    builder.add_node("router", lambda s: {})
    builder.add_node("a", lambda s: {})
    builder.add_node("b", lambda s: {})
    builder.add_conditional_edges(
        "router",
        lambda s: "x",
        {"x": "a", "y": "b"},
    )

    assert builder.successors("router") == {"a", "b"}


# ---------- Error paths --------------------------------------------------------


def test_router_returns_unknown_key_raises_at_runtime() -> None:
    class S(TypedDict):
        x: int

    builder = StateGraph(S)
    builder.add_node("n", lambda s: {})
    builder.add_edge(START, "n")
    builder.add_conditional_edges("n", lambda s: "ghost", {"a": END})
    graph = builder.compile()

    with pytest.raises(ValueError, match="ghost"):
        graph.invoke({"x": 0})


def test_double_conditional_edges_on_same_source_raises() -> None:
    class S(TypedDict):
        x: int

    builder = StateGraph(S)
    builder.add_node("n", lambda s: {})
    builder.add_conditional_edges("n", lambda s: "a", {"a": END})

    with pytest.raises(ValueError, match="conditional"):
        builder.add_conditional_edges("n", lambda s: "a", {"a": END})


def test_cannot_add_conditional_when_fixed_edge_already_exists() -> None:
    class S(TypedDict):
        x: int

    builder = StateGraph(S)
    builder.add_node("n", lambda s: {})
    builder.add_edge("n", END)

    with pytest.raises(ValueError):
        builder.add_conditional_edges("n", lambda s: "a", {"a": END})


def test_cannot_add_fixed_edge_when_conditional_already_exists() -> None:
    class S(TypedDict):
        x: int

    builder = StateGraph(S)
    builder.add_node("n", lambda s: {})
    builder.add_conditional_edges("n", lambda s: "a", {"a": END})

    with pytest.raises(ValueError):
        builder.add_edge("n", END)
