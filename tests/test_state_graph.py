"""Tests for the StateGraph builder."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

from tinygraph import END, START, StateGraph


class SimpleState(TypedDict):
    count: int


def _noop(state: SimpleState) -> dict[str, Any]:
    return {}


# ─── Basic node addition ───────────────────────────────────────


def test_add_node_stores_node() -> None:
    builder = StateGraph(SimpleState)
    builder.add_node("a", _noop)
    assert "a" in builder.nodes
    assert builder.nodes["a"] is _noop


def test_add_multiple_nodes() -> None:
    builder = StateGraph(SimpleState)
    builder.add_node("a", _noop)
    builder.add_node("b", _noop)
    assert set(builder.nodes.keys()) == {"a", "b"}


def test_add_duplicate_node_raises() -> None:
    builder = StateGraph(SimpleState)
    builder.add_node("a", _noop)
    with pytest.raises(ValueError, match="already exists"):
        builder.add_node("a", _noop)


def test_reserved_node_names_forbidden() -> None:
    """START and END are reserved and cannot be used as node names."""
    builder = StateGraph(SimpleState)
    with pytest.raises(ValueError, match="reserved"):
        builder.add_node(START, _noop)
    with pytest.raises(ValueError, match="reserved"):
        builder.add_node(END, _noop)


# ─── Basic edge addition ───────────────────────────────────────


def test_add_edge_stores_edge() -> None:
    builder = StateGraph(SimpleState)
    builder.add_node("a", _noop)
    builder.add_edge(START, "a")
    builder.add_edge("a", END)
    assert "a" in builder.successors(START)
    assert END in builder.successors("a")


def test_add_edge_to_unknown_node_raises() -> None:
    builder = StateGraph(SimpleState)
    builder.add_node("a", _noop)
    with pytest.raises(ValueError, match="Unknown"):
        builder.add_edge("a", "nonexistent")


def test_add_edge_from_unknown_node_raises() -> None:
    builder = StateGraph(SimpleState)
    builder.add_node("a", _noop)
    with pytest.raises(ValueError, match="Unknown"):
        builder.add_edge("nonexistent", "a")


def test_start_and_end_are_always_valid_endpoints() -> None:
    """START and END don't need to be `add_node`d - they're implicit."""
    builder = StateGraph(SimpleState)
    builder.add_node("a", _noop)
    # These should not raise
    builder.add_edge(START, "a")
    builder.add_edge("a", END)
