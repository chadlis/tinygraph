"""Tests for CompiledGraph.stream()."""

from typing import TypedDict

from tinygraph import END, START, StateGraph


class St(TypedDict):
    val: str


def _linear_graph() -> StateGraph[St]:
    def a(s: St) -> St:
        return {"val": s["val"] + "-A"}

    def b(s: St) -> St:
        return {"val": s["val"] + "-B"}

    g = StateGraph(St)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


def test_stream_yields_after_each_superstep() -> None:
    snapshots = list(_linear_graph().compile().stream({"val": "0"}))
    assert snapshots == [{"val": "0-A"}, {"val": "0-A-B"}]


def test_stream_last_equals_invoke() -> None:
    g = _linear_graph().compile()
    init: St = {"val": "0"}
    assert list(g.stream(init))[-1] == g.invoke(init)


def test_stream_yields_defensive_copies() -> None:
    for state in _linear_graph().compile().stream({"val": "0"}):
        state["val"] = "CORRUPTED"
    # if copies were shallow, invoke would still work but stream would leak
    assert list(_linear_graph().compile().stream({"val": "0"})) == [
        {"val": "0-A"},
        {"val": "0-A-B"},
    ]
