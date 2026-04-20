"""Tests for graph compilation and execution."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

from tinygraph import END, START, StateGraph


class CounterState(TypedDict):
    count: int


# ─── Basic execution ────────────────────────────────────────────


def test_single_node_graph() -> None:
    """Smallest possible graph: START → n → END."""

    def increment(state: CounterState) -> dict[str, Any]:
        return {"count": state["count"] + 1}

    builder = StateGraph(CounterState)
    builder.add_node("inc", increment)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)

    graph = builder.compile()
    result = graph.invoke({"count": 0})

    assert result == {"count": 1}


def test_two_node_sequential_graph() -> None:
    """START → a → b → END."""

    def double(state: CounterState) -> dict[str, Any]:
        return {"count": state["count"] * 2}

    def add_ten(state: CounterState) -> dict[str, Any]:
        return {"count": state["count"] + 10}

    builder = StateGraph(CounterState)
    builder.add_node("double", double)
    builder.add_node("add_ten", add_ten)
    builder.add_edge(START, "double")
    builder.add_edge("double", "add_ten")
    builder.add_edge("add_ten", END)

    graph = builder.compile()
    result = graph.invoke({"count": 5})

    # 5 → 10 → 20
    assert result == {"count": 20}


def test_state_is_passed_between_nodes() -> None:
    """A node can read values written by a previous node."""

    class TwoKeyState(TypedDict):
        x: int
        y: int

    def set_x(state: TwoKeyState) -> dict[str, Any]:
        return {"x": 42}

    def read_x_set_y(state: TwoKeyState) -> dict[str, Any]:
        # Must see the 42 written by set_x
        return {"y": state["x"] * 2}

    builder = StateGraph(TwoKeyState)
    builder.add_node("set_x", set_x)
    builder.add_node("read_x_set_y", read_x_set_y)
    builder.add_edge(START, "set_x")
    builder.add_edge("set_x", "read_x_set_y")
    builder.add_edge("read_x_set_y", END)

    graph = builder.compile()
    result = graph.invoke({"x": 0, "y": 0})

    assert result == {"x": 42, "y": 84}


def test_untouched_keys_preserved() -> None:
    """Keys not touched by any node must remain unchanged."""

    class MultiKeyState(TypedDict):
        a: int
        b: str

    def only_touch_a(state: MultiKeyState) -> dict[str, Any]:
        return {"a": 99}

    builder = StateGraph(MultiKeyState)
    builder.add_node("f", only_touch_a)
    builder.add_edge(START, "f")
    builder.add_edge("f", END)

    graph = builder.compile()
    result = graph.invoke({"a": 0, "b": "hello"})

    assert result == {"a": 99, "b": "hello"}


# ─── Error handling ─────────────────────────────────────────────


def test_compile_requires_start_edge() -> None:
    """A graph with no START → ... edge cannot be compiled."""
    builder = StateGraph(CounterState)
    builder.add_node("a", lambda s: {})
    # No edge from START!

    with pytest.raises(ValueError, match="START"):
        builder.compile()


def test_compile_requires_end_reachable() -> None:
    """A graph where no node points to END should error at compile time."""

    def noop(state: CounterState) -> dict[str, Any]:
        return {}

    builder = StateGraph(CounterState)
    builder.add_node("a", noop)
    builder.add_node("b", noop)
    # Chain a → b but neither points to END
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "a")  # cycle without escape
    # No edge to END!

    with pytest.raises(ValueError, match="END"):
        builder.compile()


def test_recursion_limit_exceeded() -> None:
    """The recursion limit guards against runaway execution."""

    def noop(state: CounterState) -> dict[str, Any]:
        return {}

    builder = StateGraph(CounterState)
    for i in range(10):
        builder.add_node(f"n{i}", noop)
    builder.add_edge(START, "n0")
    for i in range(9):
        builder.add_edge(f"n{i}", f"n{i + 1}")
    builder.add_edge("n9", END)

    graph = builder.compile()

    # 10 nodes + START + END = 11 steps needed, but limit is set to 5
    with pytest.raises(RecursionError):
        graph.invoke({"count": 0}, recursion_limit=5)


# ─── Structural validation ──────────────────────────────────────


def test_compile_requires_every_node_has_outgoing_edge() -> None:
    """A node registered but with no outgoing edge should be rejected."""

    def noop(state: CounterState) -> dict[str, Any]:
        return {}

    builder = StateGraph(CounterState)
    builder.add_node("a", noop)
    builder.add_node("orphan", noop)  # registered but never connected
    builder.add_edge(START, "a")
    builder.add_edge("a", END)
    # "orphan" has no outgoing edge → compile must fail

    with pytest.raises(ValueError, match="outgoing edge"):
        builder.compile()


# ─── Immutability after compile ─────────────────────────────────


def test_compiled_graph_isolated_from_builder() -> None:
    """Modifying the builder after compile() must not affect the graph."""

    def increment(state: CounterState) -> dict[str, Any]:
        return {"count": state["count"] + 1}

    builder = StateGraph(CounterState)
    builder.add_node("a", increment)
    builder.add_edge(START, "a")
    builder.add_edge("a", END)

    graph = builder.compile()

    # Now mutate the builder:
    # add a new node and redirect edges — should NOT affect compiled graph
    def double(state: CounterState) -> dict[str, Any]:
        return {"count": state["count"] * 2}

    builder.add_node("b", double)
    builder.edges[START].add("b")  # direct mutation of the builder's set

    # Compiled graph must still behave as originally compiled
    result = graph.invoke({"count": 10})
    assert result == {"count": 11}  # not 20, not something weird


# ─── Recursion limit customization ──────────────────────────────


def test_recursion_limit_can_be_increased() -> None:
    """User can pass a larger recursion_limit to handle legitimately long graphs."""

    def noop(state: CounterState) -> dict[str, Any]:
        return {}

    builder = StateGraph(CounterState)
    # Build a chain of 30 nodes: START → n0 → n1 → ... → n29 → END
    for i in range(30):
        builder.add_node(f"n{i}", noop)
    builder.add_edge(START, "n0")
    for i in range(29):
        builder.add_edge(f"n{i}", f"n{i + 1}")
    builder.add_edge("n29", END)

    graph = builder.compile()

    # Default limit is 25, so this must fail
    with pytest.raises(RecursionError):
        graph.invoke({"count": 0})

    # But with a higher limit, it should succeed
    result = graph.invoke({"count": 0}, recursion_limit=50)
    assert result == {"count": 0}
