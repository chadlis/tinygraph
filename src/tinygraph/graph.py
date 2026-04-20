from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from tinygraph.state import END, START

if TYPE_CHECKING:
    from tinygraph.state import StateGraph

MAX_STEPS: Final[int] = 25


class CompiledGraph[StateT]:
    def __init__(self, state_graph: StateGraph[StateT]) -> None:
        self._nodes = dict(state_graph.nodes)
        self._edges = {k: set(v) for k, v in state_graph.edges.items()}
        self._state_schema = state_graph.state_schema
        self._validate()

    def _validate(self) -> None:
        if (START not in self._edges) or (not self._edges[START]):
            raise ValueError("Graph has no edge from START!did you forget `add_edge(Start, ...)`?")
        for node_name in self._nodes:
            if not self._edges.get(node_name, set()):
                raise ValueError(
                    f"Node '{node_name}' has no outgoing edge"
                    f"did you forget `add_edge({node_name}, ...)`?"
                )
        if not self._is_end_reachable():
            raise ValueError("Graph has no path from START to END!")

    def _is_end_reachable(self) -> bool:
        """Return True if END is reachable from START via the edges."""
        to_visit: list[str] = list(self._edges[START])
        visited: set[str] = {START}
        while to_visit:
            current = to_visit.pop()
            if current == END:
                return True
            if current not in visited:
                visited.add(current)
                to_visit.extend(self._edges.get(current, set()))
        return False

    def invoke(self, initial_state: StateT, *, recursion_limit: int = MAX_STEPS) -> StateT:
        state: dict[str, Any] = {**cast(dict[str, Any], initial_state)}
        current = next(iter(self._edges[START]))
        for _ in range(recursion_limit):
            if current == END:
                break
            node_func = self._nodes[current]
            update = node_func(cast(StateT, state))
            state = {**state, **update}
            current = next(iter(self._edges[current]))
        else:
            raise RecursionError(f"Graph exceeded {recursion_limit} steps")
        return cast(StateT, state)
