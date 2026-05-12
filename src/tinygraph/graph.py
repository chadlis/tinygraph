from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Any, Final, cast, get_args, get_origin, get_type_hints

from tinygraph.state import END, START

if TYPE_CHECKING:
    from tinygraph.state import StateGraph

MAX_STEPS: Final[int] = 25


class CompiledGraph[StateT]:
    def __init__(self, state_graph: StateGraph[StateT]) -> None:
        self._nodes = dict(state_graph.nodes)
        self._edges = {k: set(v) for k, v in state_graph.edges.items()}
        self._conditional_edges: dict[str, tuple[Callable[[StateT], str], dict[str, str]]] = {
            k: (v[0], {**v[1]}) for k, v in state_graph.conditional_edges.items()
        }
        self._state_schema = state_graph.state_schema
        self._validate()
        self._reducers = self._get_reducers()

    def _validate(self) -> None:
        if (START not in self._edges) or (not self._edges[START]):
            raise ValueError("Graph has no edge from START! Did you forget `add_edge(Start, ...)`?")
        for node_name in self._nodes:
            if (not self._edges.get(node_name, set())) and (
                node_name not in self._conditional_edges
            ):
                raise ValueError(
                    f"Node '{node_name}' has no outgoing edge. "
                    f"Did you forget `add_edge({node_name}, ...)` "
                    f"or `add_conditional_edge({node_name}, ...)`?"
                )
        if not self._is_end_reachable():
            raise ValueError("Graph has no path from START to END!")

    def _get_reducers(self) -> dict[str, Callable[[Any, Any], Any]]:
        reducers: dict[str, Callable[[Any, Any], Any]] = {}
        type_hints = get_type_hints(self._state_schema, include_extras=True)
        for key in type_hints:
            if get_origin(type_hints[key]) is Annotated:
                reducers[key] = get_args(type_hints[key])[1]
        return reducers

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
                if current in self._edges:
                    to_visit.extend(self._edges.get(current, set()))
                if current in self._conditional_edges:
                    cond = self._conditional_edges.get(current)
                    if cond:
                        to_visit.extend(cond[1].values())
        return False

    def invoke(
        self, initial_state: StateT, *, recursion_limit: int = MAX_STEPS, current=START
    ) -> StateT:
        state: dict[str, Any] = {**cast(dict[str, Any], initial_state)}
        if recursion_limit == 0:
            raise RecursionError("Graph exceeded recursion limit")
        if current == END:
            return cast(StateT, state)
        if current != START:
            node_func = self._nodes[current]
            update = node_func(cast(StateT, state))
            self._apply_update(state, update)
        next_nodes = self._get_next_nodes(current, state)
        states = [
            {
                **cast(
                    dict[str, Any],
                    self.invoke(
                        cast(StateT, state),
                        recursion_limit=recursion_limit - 1,
                        current=node,
                    ),
                )
            }
            for node in next_nodes
        ]
        new_state = reduce(self._apply_update, states)
        if new_state is not None:
            state = new_state
        return cast(StateT, state)

    def _apply_update(self, state, update):
        for key in update:
            if key in state:
                if key in self._reducers:
                    state[key] = self._reducers[key](state[key], update[key])
                else:
                    state[key] = update[key]
        return state
        # current = self._get_next_nodes(current, state)

    def _get_next_nodes(
        self, node_name: str, state: dict[str, Any]
    ) -> set[str]:  #! TODO: typing to fix
        if node_name in self._edges:
            return self._edges[node_name]
        if node_name not in self._conditional_edges:
            raise RuntimeError(f"'{node_name}' has no fix nor conditional edges!")
        (routing_function, routing_path) = self._conditional_edges[node_name]
        routing_result = routing_function(cast(StateT, state))
        if routing_result not in routing_path:
            raise ValueError(
                f"Router returned '{routing_result}' "
                f"but it's not in the path_map of node '{node_name}'"
            )
        return set(routing_path[routing_result])
