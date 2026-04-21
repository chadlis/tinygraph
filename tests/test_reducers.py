"""Tests for Annotated reducers in state fusion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest

from tinygraph.reducers import add
from tinygraph.state import END, START, StateGraph

# ---------- Happy paths --------------------------------------------------------


def test_annotated_reducer_accumulates_instead_of_overwriting() -> None:
    """The canonical case: two nodes contribute to the same list field."""

    class ChatState(TypedDict):
        messages: Annotated[list[str], add]

    builder = StateGraph(ChatState)
    builder.add_node("a", lambda s: {"messages": ["hello"]})
    builder.add_node("b", lambda s: {"messages": ["world"]})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)

    result = builder.compile().invoke({"messages": []})
    assert result["messages"] == ["hello", "world"]


def test_field_without_reducer_is_overwritten() -> None:
    """Non-regression: plain fields keep the previous write-over behavior."""

    class S(TypedDict):
        count: int

    builder = StateGraph(S)
    builder.add_node("a", lambda s: {"count": 1})
    builder.add_node("b", lambda s: {"count": 42})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)

    result = builder.compile().invoke({"count": 0})
    assert result["count"] == 42


def test_reducer_applied_across_many_nodes() -> None:
    """The reducer composes correctly across more than two contributions."""

    class S(TypedDict):
        items: Annotated[list[int], add]

    def _make_node(i: int) -> Callable[[S], dict[str, list[int]]]:
        return lambda s: {"items": [i]}

    builder = StateGraph(S)
    for i in range(1, 4):
        builder.add_node(f"n{i}", _make_node(i))
    builder.add_edge(START, "n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", "n3")
    builder.add_edge("n3", END)

    result = builder.compile().invoke({"items": []})
    assert result["items"] == [1, 2, 3]


def test_mixed_fields_reducer_and_plain_coexist() -> None:
    """A state can mix fields with and without reducers in a single schema."""

    class S(TypedDict):
        log: Annotated[list[str], add]
        status: str

    builder = StateGraph(S)
    builder.add_node("a", lambda s: {"log": ["step-a"], "status": "running"})
    builder.add_node("b", lambda s: {"log": ["step-b"], "status": "done"})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)

    result = builder.compile().invoke({"log": [], "status": "init"})
    assert result["log"] == ["step-a", "step-b"]
    assert result["status"] == "done"


def test_field_absent_from_update_is_untouched() -> None:
    """A node that omits a key leaves that field unchanged in the state."""

    class S(TypedDict):
        messages: Annotated[list[str], add]
        meta: str

    builder = StateGraph(S)
    builder.add_node("a", lambda s: {"messages": ["hi"]})  # does NOT touch `meta`
    builder.add_edge(START, "a")
    builder.add_edge("a", END)

    result = builder.compile().invoke({"messages": [], "meta": "kept"})
    assert result["meta"] == "kept"
    assert result["messages"] == ["hi"]


# ---------- Reducer contract ---------------------------------------------------


def test_add_reducer_raises_on_unsupported_type() -> None:
    """`add` only supports lists; other types must fail loudly."""
    with pytest.raises(TypeError):
        add(42, 1)
