"""Gantt chart example."""


from typing import Iterable, Dict, Optional, Union, Callable
from collections import defaultdict

Annots = Iterable[Dict[str, Union[str, int, float]]]


def _task_lines(annots: Annots):
    for a in annots:
        yield f"    \"{a['task']}\" : {a['bt']}, {a['tt']}"


def mermaid_gantt(
    annots: Annots,
    *,
    title: Optional[str] = None,
    date_format: str = "s",
    axis_format: Optional[str] = None,
    group: Optional[Callable] = None,
) -> str:
    """
    Creates a Mermaid Gantt diagram string from the given annotations.

    :param annots: Iterable of annotations (dicts with "task", "bt", and "tt" keys).
    :param title: Optional title for the Gantt diagram.
    :param date_format: Date format for the Gantt diagram. Default is 's' (seconds).
        See https://mermaid.js.org/syntax/gantt.html#input-date-format for more info.
    :param axis_format: Axis format for the Gantt diagram. Default is None (no axis).
        See https://mermaid.js.org/syntax/gantt.html#output-date-format-on-the-axis for more info
    :return: A string in Mermaid Gantt diagram syntax.

    >>> annots = [
    ...     {"task": "snap", "bt": 1, "tt": 4},
    ...     {"task": "crackle", "bt": 5, "tt": 8},
    ...     {"task": "pop", "bt": 9, "tt": 12},
    ...     {"task": "snap", "bt": 3, "tt": 7},
    ... ]
    >>> mermaid_str = mermaid_gantt(
    ...     annots,
    ...     title="Example Gantt Diagram",
    ...     date_format='s',
    ... )
    >>>
    >>> print(mermaid_str)
    gantt
        dateFormat  s
        axisFormat  %S
        title Example Gantt Diagram
        "snap" : 1, 4
        "crackle" : 5, 8
        "pop" : 9, 12
        "snap" : 3, 7


    You can group tasks by a function of the annotation dict.
    For example, to group by the task name itself:

    >>> mermaid_str_2 = mermaid_gantt(
    ...     annots,
    ...     title="Example Gantt Diagram",
    ...     date_format='s',
    ...     group=lambda x: x['task']
    ... )
    >>> print(mermaid_str_2)
    gantt
        dateFormat  s
        axisFormat  %S
        title Example Gantt Diagram
        Section snap
        "snap" : 1, 4
        "snap" : 3, 7
        Section crackle
        "crackle" : 5, 8
        Section pop
        "pop" : 9, 12
    <BLANKLINE>
    """

    # Start the Gantt diagram syntax
    mermaid_str = "gantt\n"

    # Set the date format
    mermaid_str += f"    dateFormat  {date_format}\n"

    if axis_format is None:
        annots = list(annots)  # to not consume, and lose, the iterator
        axis_format = mk_axis_format(date_format, annots)
    if axis_format is not None:
        mermaid_str += f"    axisFormat  {axis_format}\n"

    # Set the title if given
    if title:
        mermaid_str += f"    title {title}\n"

    if group is None:
        mermaid_str += "\n".join(_task_lines(annots))
    else:
        grouped_annots = defaultdict(list)
        for annot in annots:
            grouped_annots[group(annot)].append(annot)
        for group, group_annots in grouped_annots.items():
            mermaid_str += f"    Section {group}\n"
            mermaid_str += "\n".join(_task_lines(group_annots))
            mermaid_str += "\n"

    # Return the final Mermaid Gantt diagram string
    return mermaid_str


# TODO: Not using annots here -- see below for range-based approach
def mk_axis_format(date_format, annots):
    return date_to_axis_format_mapping.get(date_format)


date_to_axis_format_mapping = {
    "YYYY": "%Y",
    "YY": "%y",
    "Q": "%m",
    "M": "%m",
    "MM": "%m",
    "MMM": "%b",
    "MMMM": "%B",
    "D": "%d",
    "DD": "%d",
    "Do": "%d",
    "DDD": "%j",
    "DDDD": "%j",
    "X": "%s",
    "x": "%s",
    "H": "%H",
    "HH": "%H",
    "h": "%I",
    "hh": "%I",
    "a": "%p",
    "A": "%p",
    "m": "%M",
    "mm": "%M",
    "s": "%S",
    "ss": "%S",
    "S": "%L",
    "SS": "%L",
    "SSS": "%L",
    "Z": "%Z",
    "ZZ": "%Z",
}


# -------------------------------------------------------------------------------------
# Some scrap (not used... yet) on getting the axis format from the range expressed
# by the annots.
#
# From a specification of `dateFormat`,  I'd like to compute a good default for `axisFormat`.
# A good option would be one that shows the right level of detail of the axis ticks,
# without any tick labels being the same.
# To do this, I suggest you create a mapping matching as many of the options for
# dateFormat and axisFormat as possible, then use that mapping to construct an
# axisFormat based on the dataFormat as well as the total range of the gantt diagram
# (which you can compute from the `annots` input argument.

from collections.abc import Callable


class IntervalSwitchCase(Callable):
    def __init__(self, mapping, default):
        # Store mapping as a sorted tuple
        self.mapping = tuple(sorted(mapping, reverse=True))
        self.default = default

    def __call__(self, value):
        # Implement switch case logic
        for threshold, result in self.mapping:
            if value > threshold:
                return result
        return self.default

    def update(self, updates):
        # Immutably update mapping
        updated_mapping = dict(self.mapping)
        updated_mapping.update(updates)
        return IntervalSwitchCase(list(updated_mapping.items()), self.default)

    def __setitem__(self, key, value):
        # Support item assignment syntax
        updated_mapping = dict(self.mapping)
        updated_mapping[key] = value
        return IntervalSwitchCase(list(updated_mapping.items()), self.default)

    def __delitem__(self, key):
        # Support item deletion syntax
        updated_mapping = dict(self.mapping)
        del updated_mapping[key]
        return IntervalSwitchCase(list(updated_mapping.items()), self.default)


# Define default_mapping and default for axis_format case
axis_format_default_mapping = (
    (365, "%Y"),
    (30, "%Y-%m"),
    (7, "%Y-%m-%d"),
    (1, "%Y-%m-%d %H:%M"),
    (0.041666666666666664, "%Y-%m-%d %H:%M:%S"),
    (0.0006944444444444444, "%Y-%m-%d %H:%M:%S.%L"),
    (1.1574074074074073e-05, "%Y-%m-%d %H:%M:%S.%L"),
)
axis_format_default = "%Y-%m-%d %H:%M:%S.%L"

# Create a IntervalSwitchCase instance for the axis_format
axis_format_selector = IntervalSwitchCase(
    axis_format_default_mapping, axis_format_default
)

# # Example usage
# print(axis_format_selector(50))  # Output: '%Y-%m'

# # Get an updated IntervalSwitchCase object with custom mapping
# custom_selector = axis_format_selector.update({1: '%Y-%m-%d %H'})
# print(custom_selector(0.5))  # Output: '%Y-%m-%d %H'


def _get_axis_format(date_format, total_range_in_days):
    """
    Computes an appropriate axisFormat string based on the dateFormat string and the total range of the Gantt diagram.

    :param date_format: A string specifying the date format in Mermaid syntax.
    :param total_range: The total range of the Gantt diagram.
    :return: A string representing the axisFormat.
    """

    tokens = date_format.split()
    axis_format = ""

    # Get the level of detail in the date_format
    for token in tokens:
        if token in date_to_axis_format_mapping:
            axis_format += date_to_axis_format_mapping[token] + " "

    # If the total range is large, we might want to reduce the detail
    if total_range_in_days > 365:  # More than a year
        axis_format = "%Y"
    elif total_range_in_days > 30:  # More than a month
        axis_format = "%Y-%m"
    elif total_range_in_days > 7:  # More than a week
        axis_format = "%Y-%m-%d"
    elif total_range_in_days > 1:  # More than a day
        axis_format = "%Y-%m-%d %H:%M"
    elif total_range_in_days > 0.041666666666666664:  # More than an hour
        axis_format = "%Y-%m-%d %H:%M:%S"
    elif total_range_in_days > 0.0006944444444444444:  # More than a minute
        axis_format = "%Y-%m-%d %H:%M:%S.%L"
    elif total_range_in_days > 0.000011574074074074073:  # More than a second
        axis_format = "%Y-%m-%d %H:%M:%S.%L"
    else:  # Less than a second
        axis_format = "%Y-%m-%d %H:%M:%S.%L"

    return axis_format.strip()
