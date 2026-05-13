from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from tinygraph.checkpoint import BaseCheckpointer
from tinygraph.state import END, START

if TYPE_CHECKING:
    from tinygraph.state import StateGraph

MAX_STEPS: Final[int] = 25


class CompiledGraph[StateT]:
    def __init__(
        self, state_graph: StateGraph[StateT], checkpointer: BaseCheckpointer | None = None
    ) -> None:
        self._nodes = dict(state_graph.nodes)
        self._edges = {k: set(v) for k, v in state_graph.edges.items()}
        self._conditional_edges: dict[str, tuple[Callable[[StateT], str], dict[str, str]]] = {
            k: (v[0], {**v[1]}) for k, v in state_graph.conditional_edges.items()
        }
        self._state_schema = state_graph.state_schema
        self._validate()
        self._reducers = self._get_reducers()
        self._predecessors = self._build_predecessors()
        self._checkpointer = checkpointer

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

    def _build_predecessors(self) -> dict[str, set[str]]:
        predecessors: defaultdict[str, set[str]] = defaultdict(set)
        for source, targets in self._edges.items():
            for target in targets:
                predecessors[target].add(source)
        for source, (_, path_map) in self._conditional_edges.items():
            for target in path_map.values():
                predecessors[target].add(source)
        return dict(predecessors)

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
                    to_visit.extend(self._edges[current])
                if current in self._conditional_edges:
                    cond = self._conditional_edges.get(current)
                    if cond:
                        to_visit.extend(cond[1].values())
        return False

    def _run_steps(
        self, initial_state: StateT, *, recursion_limit: int = MAX_STEPS
    ) -> Iterator[StateT]:
        state: dict[str, Any] = {**cast(dict[str, Any], initial_state)}
        completed: set[str] = {START}
        activated: set[str] = {START}
        frontier = self._next_frontier({START}, completed, activated, state)
        activated |= frontier

        for _ in range(recursion_limit):
            if END in frontier:
                return

            snapshot = dict(state)
            updates: list[dict[str, Any]] = []
            for node_name in frontier:
                node_func = self._nodes[node_name]
                updates.append(node_func(cast(StateT, snapshot)))

            for update in updates:
                self._apply_update(state, update)

            completed |= frontier

            yield cast(StateT, {**state})

            frontier = self._next_frontier(frontier, completed, activated, state)
            activated |= frontier

            if not frontier:
                raise RuntimeError("Graph stuck: no nodes ready and END not reached.")

        raise RecursionError(f"Graph exceeded {recursion_limit} steps")

    def invoke(
        self,
        initial_state: StateT,
        *,
        config: dict[str, str] | None = None,
        recursion_limit: int = MAX_STEPS,
    ) -> StateT:
        state = {**cast(dict[str, Any], initial_state)}

        if config is not None and self._checkpointer is not None:
            thread_id = config["thread_id"]
            saved = self._checkpointer.get(thread_id)
            if saved is not None:
                state = dict(saved)
                self._apply_update(state, cast(dict[str, Any], initial_state))

        result: StateT = cast(StateT, state)
        for step in self._run_steps(cast(StateT, state), recursion_limit=recursion_limit):
            result = step

        if config is not None and self._checkpointer is not None:
            self._checkpointer.put(config["thread_id"], {**cast(dict[str, Any], result)})

        return result

    def stream(
        self, initial_state: StateT, *, recursion_limit: int = MAX_STEPS
    ) -> Iterator[StateT]:
        return self._run_steps(initial_state, recursion_limit=recursion_limit)

    def _next_frontier(
        self,
        current_frontier: set[str],
        completed: set[str],
        activated: set[str],
        state: dict[str, Any],
    ) -> set[str]:
        candidates: set[str] = set()
        for node_name in current_frontier:
            candidates |= self._get_next_nodes(node_name, state)
        barrier_scope = activated | candidates
        return {
            n for n in candidates if (self._predecessors.get(n, set()) & barrier_scope) <= completed
        }

    def _apply_update(self, state: dict[str, Any], update: dict[str, Any]) -> None:
        for key in update:
            if key in state:
                if key in self._reducers:
                    state[key] = self._reducers[key](state[key], update[key])
                else:
                    state[key] = update[key]

    def _get_next_nodes(self, node_name: str, state: dict[str, Any]) -> set[str]:
        if node_name in self._edges:
            return self._edges[node_name]
        if node_name not in self._conditional_edges:
            raise RuntimeError(f"'{node_name}' has no fixed nor conditional edges!")
        (routing_function, routing_path) = self._conditional_edges[node_name]
        routing_result = routing_function(cast(StateT, state))
        if routing_result not in routing_path:
            raise ValueError(
                f"Router returned '{routing_result}' "
                f"but it's not in the path_map of node '{node_name}'"
            )
        return {routing_path[routing_result]}
