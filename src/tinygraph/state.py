# Note: edges are stored as sets (fast dedup + O(1) lookup).
# If we later need deterministic execution order across parallel
# branches, switching to lists may be needed

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Final

START: Final[str] = "__start__"
END: Final[str] = "__end__"


class StateGraph[StateT]:
    """A builder for declaring stateful computation graphs.

    Example:
        >>> class MyState(TypedDict):
        ...     count: int
        >>> builder = StateGraph(MyState)
        >>> builder.add_node("increment", lambda s: {"count": s["count"] + 1})
        >>> builder.add_edge(START, "increment")
        >>> builder.add_edge("increment", END)
    """

    def __init__(self, state_schema: type[StateT]) -> None:
        self.state_schema: type[StateT] = state_schema
        self.nodes: dict[str, Callable[[StateT], dict[str, Any]]] = {}
        self.edges: defaultdict[str, set[str]] = defaultdict(set)

    def add_node(self, name: str, node: Callable[[StateT], dict[str, Any]]) -> None:
        """Register a node function under `name`.

        Raises:
            ValueError: if `name` is reserved or already registered.
        """
        if name in (START, END):
            raise ValueError(f"'{name}' is a reserved node name")
        if name in self.nodes:
            raise ValueError(f"'{name}' already exists")
        self.nodes[name] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge between two nodes.

        Raises:
            ValueError: if either endpoint is an unknown node
                (START and END are always valid).
        """
        if from_node != START and from_node not in self.nodes:
            raise ValueError(f"Unknown source node: {from_node}")
        if to_node != END and to_node not in self.nodes:
            raise ValueError(f"Unknown target node: {to_node}")
        self.edges[from_node].add(to_node)

    def successors(self, name: str) -> set[str]:
        """Return the direct successors of `name` (empty set if none)."""
        return self.edges.get(name, set())
