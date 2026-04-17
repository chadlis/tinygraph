"""Smoke test: make sure the package can be imported."""
from __future__ import annotations


def test_package_imports() -> None:
    import tinygraph

    assert tinygraph.__version__


def test_version_is_string() -> None:
    import tinygraph

    assert isinstance(tinygraph.__version__, str)
