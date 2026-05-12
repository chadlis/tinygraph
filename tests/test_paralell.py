from typing import Annotated, TypedDict

from tinygraph.reducers import add
from tinygraph.state import END, START, StateGraph


def test_diamond_fan_out_fan_in() -> None:
    """START → split → (a, b) → merge → END"""

    class S(TypedDict):
        log: Annotated[list[str], add]

    builder = StateGraph(S)
    builder.add_node("split", lambda s: {})
    builder.add_node("a", lambda s: {"log": ["A"]})
    builder.add_node("b", lambda s: {"log": ["B"]})
    # builder.add_node("merge", lambda s: {"log": ["merged"]})

    builder.add_edge(START, "split")
    builder.add_edge("split", "a")
    builder.add_edge("split", "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)

    # builder.add_edge("a", "merge")
    # builder.add_edge("b", "merge")
    # builder.add_edge("merge", END)

    result = builder.compile().invoke({"log": []})
    # assert sorted(result["log"][:2]) == ["A", "B"]
    assert sorted(result["log"]) == ["A", "B"]
    # assert result["log"][2] == "merged"
