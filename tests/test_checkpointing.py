"""Tests for checkpointing."""

from typing import Annotated, Any, TypedDict

from tinygraph import END, START, StateGraph
from tinygraph.checkpoint import MemorySaver


class Chat(TypedDict):
    history: Annotated[list[str], lambda old, new: old + new]


def echo(state: Chat) -> dict[str, Any]:
    return {"history": [f"echo: {state['history'][-1]}"]}


def _build() -> StateGraph[Chat]:
    g = StateGraph(Chat)
    g.add_node("echo", echo)
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    return g


def test_invoke_without_checkpointer_unchanged() -> None:
    g = _build().compile()
    result = g.invoke({"history": ["hi"]})
    assert result == {"history": ["hi", "echo: hi"]}


def test_two_invokes_same_thread_accumulate() -> None:
    g = _build().compile(checkpointer=MemorySaver())
    cfg = {"thread_id": "t1"}

    r1 = g.invoke({"history": ["hello"]}, config=cfg)
    r2 = g.invoke({"history": ["world"]}, config=cfg)

    assert r1 == {"history": ["hello", "echo: hello"]}
    assert r2 == {"history": ["hello", "echo: hello", "world", "echo: world"]}


def test_different_threads_are_isolated() -> None:
    g = _build().compile(checkpointer=MemorySaver())

    g.invoke({"history": ["aaa"]}, config={"thread_id": "t1"})
    r = g.invoke({"history": ["bbb"]}, config={"thread_id": "t2"})

    assert r == {"history": ["bbb", "echo: bbb"]}
