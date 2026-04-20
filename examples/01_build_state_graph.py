from typing import TypedDict

from tinygraph import END, START, StateGraph


class MyState(TypedDict):
    count: int
    message: str


def node_a(state: MyState) -> dict:
    return {"count": state["count"] + 1}


def node_b(state: MyState) -> dict:
    return {"message": "hello"}


# --- L'API que tu dois implémenter : ---
builder = StateGraph(MyState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)
