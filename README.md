# allude
Building precise functionality from vague specifications

To install:	```pip install allude```

`allude` collects small tools that turn loose, human-friendly specifications (a list of annotations, a dot graph, "this is proportional to that") into precise artifacts: diagram source, edge lists and callable functions with meaningful signatures.

## Examples

### A Gantt chart from a list of annotations

Give `mermaid_gantt` an iterable of dicts with a task name, a begin time (`bt`) and an end time (`tt`), and it gives you [Mermaid](https://mermaid.js.org/syntax/gantt.html) Gantt source that you can paste into any Mermaid renderer (GitHub markdown, mermaid.live, ...):

```python
>>> from allude import mermaid_gantt
>>> annots = [
...     {"task": "design", "bt": 0, "tt": 3},
...     {"task": "build", "bt": 3, "tt": 10},
...     {"task": "test", "bt": 8, "tt": 12},
... ]
>>> print(mermaid_gantt(annots, title="Project"))
gantt
    dateFormat  s
    axisFormat  %S
    title Project
    "design" : 0, 3
    "build" : 3, 10
    "test" : 8, 12

```

Group tasks into sections with any function of the annotation:

```python
>>> annots = [
...     {"task": "design", "bt": 0, "tt": 3, "team": "ux"},
...     {"task": "build", "bt": 3, "tt": 10, "team": "eng"},
...     {"task": "user tests", "bt": 8, "tt": 12, "team": "ux"},
... ]
>>> print(mermaid_gantt(annots, group=lambda a: a["team"]))
gantt
    dateFormat  s
    axisFormat  %S
    Section ux
    "design" : 0, 3
    "user tests" : 8, 12
    Section eng
    "build" : 3, 10
<BLANKLINE>

```

If your dicts use other field names, say which ones with the `task=`, `bt=` and `tt=` arguments.

### Edges from a graphviz (dot) specification

Sketch a graph in dot syntax (a string, or a `graphviz` object) and get its edges as `(source, target)` pairs:

```python
>>> from allude.util import get_edges
>>> get_edges('''
... digraph {
...     rain -> traffic
...     rain -> wet
...     traffic -> moody
...     wet -> moody
... }
... ''')
[('rain', 'traffic'), ('rain', 'wet'), ('traffic', 'moody'), ('wet', 'moody')]

```

### Functions from "this is proportional to that"

`mk_proportional_change_func` makes a function that updates a `prior` in proportion to some `evidence`, and gives it a name and a signature that use your own variable names:

```python
>>> import inspect
>>> from allude.util import mk_proportional_change_func
>>> rain_to_mood = mk_proportional_change_func("mood", "rain", 0.5)
>>> rain_to_mood.__name__
'rain_to_mood'
>>> print(inspect.signature(rain_to_mood))
(mood, rain, *, mood_rain=0.5)
>>> rain_to_mood(10, 2)  # mood * (rain * 0.5)
10.0
>>> rain_to_mood(mood=10, rain=2, mood_rain=0.1)
2.0

```

Such functions support arithmetic, so you can scale them:

```python
>>> (rain_to_mood * 3)(10, 2)
30.0

```
