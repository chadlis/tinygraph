from typing import Any


def add(current: Any, update: Any) -> Any:
    if isinstance(current, list):
        return [*current, *update]
    raise TypeError(f"unsupported type : {type(current)} for {add.__name__}")
