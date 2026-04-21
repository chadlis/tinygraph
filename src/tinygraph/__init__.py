"""tinygraph —  a minimalist implementation of LangGraph-like core."""

from tinygraph.graph import CompiledGraph
from tinygraph.reducers import add
from tinygraph.state import END, START, StateGraph

__version__ = "0.1.0"
__all__ = ["END", "START", "CompiledGraph", "StateGraph", "add"]
