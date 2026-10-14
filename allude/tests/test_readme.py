"""Run the README's examples as doctests, so the documented usage stays true."""

import doctest
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_examples():
    if not README.exists():  # e.g. when running from an installed wheel
        return
    results = doctest.testfile(
        str(README),
        module_relative=False,
        verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )
    assert results.failed == 0, f"{results.failed} README example(s) failed"
