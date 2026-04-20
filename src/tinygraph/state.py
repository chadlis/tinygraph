# Note: edges are stored as sets (fast dedup + O(1) lookup).
# If we later need deterministic execution order across parallel
# branches, switching to lists may be needed
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from tinygraph.graph import CompiledGraph

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
        self.conditional_edges: dict[str, tuple[Callable[[StateT], str], dict[str, str]]] = {}

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
        if from_node == END:
            raise ValueError(f"Non valid value {END} for source node")
        if to_node == START:
            raise ValueError(f"Non valid value {START} for target node")
        if from_node != START and from_node not in self.nodes:
            raise ValueError(f"Unknown source node: {from_node}")
        if to_node != END and to_node not in self.nodes:
            raise ValueError(f"Unknown target node: {to_node}")
        #! TODO : remove it after parallel fan-out is impelemented
        if self.edges.get(from_node):
            raise ValueError(
                f"Node '{from_node}' already has an outgoing edge. "
                "Use `add_conditional_edges()` for branching."
            )
        if self.conditional_edges.get(from_node):
            raise ValueError(f"Node '{from_node}' already has a conditional edge.")
        self.edges[from_node].add(to_node)

    def add_conditional_edges(
        self, from_node: str, path: Callable[[StateT], str], path_map: dict[str, str]
    ) -> None:
        if from_node == END:
            raise ValueError("Cannot add conditional edges from END")
        if from_node not in self.nodes:
            raise ValueError(
                f"Node '{from_node}' is not registered! Did you forget to call `add_node()`?"
            )
        for node_name in path_map.values():
            if (node_name != END) and (node_name not in self.nodes):
                raise ValueError(
                    f"Node '{node_name}' is not registered! Did you forget to call `add_node()`?"
                )
        if len(self.edges.get(from_node, set())) > 0:
            raise ValueError(f"Node '{from_node}' has already one or more fix edges!")
        if self.conditional_edges.get(from_node):
            raise ValueError(f"Node '{from_node}' already has a conditional edge.")
        self.conditional_edges[from_node] = (path, path_map)

    def successors(self, name: str) -> set[str]:
        """Return the direct successors of `name` (empty set if none)."""
        cond = self.conditional_edges.get(name)
        if cond:
            return self.edges.get(name, set()) | set(cond[1].values())
        return self.edges.get(name, set())

    def compile(self) -> CompiledGraph[StateT]:
        """Compile this builder into an executable graph.

        Raises:
            ValueError: if the graph is invalid (no START edge, END unreachable).
        """
        from tinygraph.graph import CompiledGraph

        return CompiledGraph(self)
