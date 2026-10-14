# allude.util

Utils

### Functions

| [`edge_to_node_ids`](#allude.util.edge_to_node_ids)(edge)                          | Returns the node id pair for the edge object                                                |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `func_op_func`(op, func1, func2)                                                                 |                                                                                             |
| `func_op_obj`(op, func, obj)                                                                     |                                                                                             |
| [`get_edges`](#allude.util.get_edges)(graph_spec[, postprocess_edges])      | Get a list of edges for a given graph (or list of lists thereof).                           |
| [`get_graph_dot_obj`](#allude.util.get_graph_dot_obj)(graph_spec)                   | Get a dot (graphs) object list from a variety of possible sources (postelizing inputs here) |
| [`mk_proportional_change_func`](#allude.util.mk_proportional_change_func)(prior_argname, ...) | Make a `proportional_change` function whose arguments carry your own names.                 |
| `proportional_change`(prior, evidence, proportion)                                               |                                                                                             |

### Classes

| `FuncWrap`(func)     |    |
|----------------------|----|
| `OperableFunc`(func) |    |

### allude.util.edge_to_node_ids(edge)

Returns the node id pair for the edge object

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### allude.util.get_edges(graph_spec, postprocess_edges=<function edge_to_node_ids>)

Get a list of edges for a given graph (or list of lists thereof).
If `postprocess_edges` is `None` the function will return `pydot.Edge` objects from
which you can extract any information you want.
By default though, it is set to extract the node pairs for the edges, and you can
replace with any function that takes `pydot.Edge` as an input.

```pycon
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
```

### allude.util.get_graph_dot_obj(graph_spec)

Get a dot (graphs) object list from a variety of possible sources (postelizing inputs here)

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`Dot`]

### allude.util.mk_proportional_change_func(prior_argname, evidence_argname, proportion, proportion_argname=None, name=None)

Make a `proportional_change` function whose arguments carry your own names.

The returned function computes `prior * (evidence * proportion)`, where
`proportion` is a keyword-only argument defaulting to `proportion`.

```pycon
>>> rain_to_mood = mk_proportional_change_func("mood", "rain", 0.5)
>>> rain_to_mood.__name__
'rain_to_mood'
>>> rain_to_mood(10, 2)
10.0
>>> rain_to_mood(mood=10, rain=2, mood_rain=0.1)
2.0
```
