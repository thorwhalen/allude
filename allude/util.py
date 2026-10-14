"""Utils"""

import inspect
from operator import mul, methodcaller
from collections.abc import Callable

from i2 import Sig
from lined import Pipe, map_star
from lined.util import func_name
from meshed import DAG
from meshed.util import func_name

try:
    import importlib.resources

    _files = importlib.resources.files  # only valid in 3.9+
except AttributeError:
    import importlib_resources  # needs pip install

    _files = importlib_resources.files

files = _files("allude")
data_path = files / "data"
data_dir = str(data_path)


class FuncWrap:
    def __init__(self, func):
        self.func = func
        Sig(func)(self)  # make self have func's signature
        self.__name__ = func_name(func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"FuncWrap({repr(self.func)})"


# def func_op_obj(*args, op, func, obj, **kwargs):
#     return op(func(*args, **kwargs), obj)


def func_op_obj(op, func, obj):
    obj_op = lambda x: op(x, obj)  # Pattern: Extended partial
    return Pipe(func, obj_op)


def func_op_func(
    op,
    func1,
    func2,
):
    return Pipe(DAG([func1, func2]), methodcaller("values"), map_star(op))


class OperableFunc(FuncWrap):
    def __mul__(self, other):
        if isinstance(other, Callable):
            new_func = func_op_func(op=mul, func1=self.func, func2=other)
            # new_func = partial(func_op_func, op=mul, func1=self.func, func2=other)
        else:
            new_func = func_op_obj(op=mul, func=self.func, obj=other)
            # new_func = partial(func_op_obj, op=mul, func=self.func, obj=other)
        # Sig(self)(new_func)  # give new_func the signature of self

        return OperableFunc(new_func)


def proportional_change(prior, evidence, proportion):
    return prior * (evidence * proportion)


class _RenamedProportionalChange:
    """``proportional_change`` exposed under a renamed signature (picklable)."""

    def __init__(self, sig, argnames, name):
        self.__signature__ = sig
        self._argnames = argnames
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        return proportional_change(*(bound.arguments[k] for k in self._argnames))


def mk_proportional_change_func(
    prior_argname,
    evidence_argname,
    proportion,
    proportion_argname: str = None,
    name: str = None,
):
    """Make a ``proportional_change`` function whose arguments carry your own names.

    The returned function computes ``prior * (evidence * proportion)``, where
    ``proportion`` is a keyword-only argument defaulting to ``proportion``.

    >>> rain_to_mood = mk_proportional_change_func("mood", "rain", 0.5)
    >>> rain_to_mood.__name__
    'rain_to_mood'
    >>> rain_to_mood(10, 2)
    10.0
    >>> rain_to_mood(mood=10, rain=2, mood_rain=0.1)
    2.0
    """
    name = name or f"{evidence_argname}_to_{prior_argname}"
    proportion_argname = proportion_argname or f"{prior_argname}_{evidence_argname}"
    P = inspect.Parameter
    sig = inspect.Signature(
        [
            P(prior_argname, P.POSITIONAL_OR_KEYWORD),
            P(evidence_argname, P.POSITIONAL_OR_KEYWORD),
            P(proportion_argname, P.KEYWORD_ONLY, default=proportion),
        ]
    )
    _proportional_change = _RenamedProportionalChange(
        sig, (prior_argname, evidence_argname, proportion_argname), name
    )
    func = OperableFunc(_proportional_change)
    func.__name__ = name
    return func


# ---------------------------------------------------------------------------------------
# Extract elements from graphviz objects

from typing import Tuple, List
from collections.abc import Iterable
from pydot import Dot, graph_from_dot_data, Edge
from graphviz.graphs import BaseGraph
from graphviz import Source


def edge_to_node_ids(edge: Edge) -> tuple[str, str]:
    """Returns the node id pair for the edge object"""
    return edge.get_source(), edge.get_destination()


def get_graph_dot_obj(graph_spec) -> list[Dot]:
    """Get a dot (graphs) object list from a variety of possible sources (postelizing inputs here)"""
    _original_graph_spec = graph_spec
    if isinstance(graph_spec, (BaseGraph, Source)):
        # get the source (str) from a graph object
        graph_spec = graph_spec.source
    if isinstance(graph_spec, str):
        # get a dot-graph from dot string data
        graph_spec = graph_from_dot_data(graph_spec)
    # make sure we have a list of Dot objects now
    assert isinstance(graph_spec, list) and all(
        isinstance(x, Dot) for x in graph_spec
    ), (
        f"Couldn't get a proper dot object list from: {_original_graph_spec}. "
        f"At this point, we should have a list of Dot objects, but was: {graph_spec}"
    )
    return graph_spec


def get_edges(graph_spec, postprocess_edges=edge_to_node_ids):
    r"""Get a list of edges for a given graph (or list of lists thereof).
    If ``postprocess_edges`` is ``None`` the function will return ``pydot.Edge`` objects from
    which you can extract any information you want.
    By default though, it is set to extract the node pairs for the edges, and you can
    replace with any function that takes ``pydot.Edge`` as an input.

    >>> digraph_dot_source = '''
    ... digraph {
    ...     rain -> traffic
    ...     rain -> wet
    ...     traffic -> moody
    ...     wet -> moody
    ... }
    ... '''
    >>> assert (
    ...     get_edges(digraph_dot_source)
    ...     == get_edges(Source(digraph_dot_source))
    ...     == [('rain', 'traffic'), ('rain', 'wet'), ('traffic', 'moody'), ('wet', 'moody')]
    ... )
    >>>
    >>> graph_dot_source = '''
    ... graph {
    ...     rain -- traffic
    ...     rain -- wet
    ...     traffic -- moody
    ...     wet -- moody
    ... }
    ... '''
    >>>
    >>> assert (
    ...     get_edges(graph_dot_source)
    ...     == get_edges(Source(graph_dot_source))
    ...     == [('rain', 'traffic'), ('rain', 'wet'), ('traffic', 'moody'), ('wet', 'moody')]
    ... )
    """
    graphs = get_graph_dot_obj(graph_spec)
    n_graphs = len(graphs)

    if n_graphs > 1:
        return [get_edges(graph, postprocess_edges) for graph in graphs]
    elif n_graphs == 0:
        raise ValueError(f"Your input had no graphs")
    else:
        graph = graphs[0]
        edges = graph.get_edges()
        if callable(postprocess_edges):
            edges = list(map(postprocess_edges, edges))
        return edges
