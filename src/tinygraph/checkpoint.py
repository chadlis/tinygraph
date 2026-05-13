from typing import Any, Protocol


class BaseCheckpointer(Protocol):
    def get(self, thread_id: str) -> dict[str, Any] | None: ...

    def put(self, thread_id: str, state: dict[str, Any]) -> None: ...


class MemorySaver:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def get(self, thread_id: str) -> dict[str, Any] | None:
        return self._store.get(thread_id)

    def put(self, thread_id: str, state: dict[str, Any]) -> None:
        self._store[thread_id] = state
