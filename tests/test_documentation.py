"""The documentation names real things.

Prose goes stale silently. A renamed parameter or a moved class leaves a
tutorial that still reads correctly and no longer works, and the only reader
who finds out is the one following it. Every ``pyhighlights`` import written in
the README or under ``docsrc/source`` is resolved here, so a rename either
updates the docs or fails the suite.
"""

import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: ``from pyhighlights.x import a, b`` and its parenthesized, multi-line form.
IMPORT = re.compile(
    r"from\s+(pyhighlights[\w.]*)\s+import\s+(?:\(([^)]*)\)|([^\n(]+))",
)


def documented() -> list:
    """Every name every documentation file imports from the package."""
    sources = [ROOT / "README.md", *sorted((ROOT / "docsrc" / "source").rglob("*.rst"))]
    found = []
    for source in sources:
        text = source.read_text()
        for match in IMPORT.finditer(text):
            module = match.group(1)
            names = match.group(2) or match.group(3) or ""
            for name in names.replace("\n", " ").split(","):
                name = name.strip().split(" as ")[0].strip()
                # A trailing `#: comment` or an ellipsis in an elided list.
                if name and name.isidentifier():
                    found.append((source.relative_to(ROOT), module, name))
    return found


def test_the_documentation_imports_things_that_exist():
    references = documented()
    # A guard that finds nothing guards nothing.
    assert len(references) > 20

    missing = []
    for source, module, name in references:
        try:
            imported = importlib.import_module(module)
        except ImportError:  # pragma: no cover - an optional extra
            continue
        if not hasattr(imported, name):
            missing.append(f"{source}: {module}.{name}")

    assert not missing, missing


#: The body of a ``toctree`` directive: indented lines up to the next
#: unindented one.
TOCTREE = re.compile(r"\.\. toctree::\n(?:   :[\w-]+:.*\n)*\n((?:(?:   \S.*)?\n)*)")


def test_every_page_the_index_lists_is_there():
    """A toctree entry with no file behind it is a broken build."""
    source = ROOT / "docsrc" / "source"
    listed = [
        page
        for block in TOCTREE.findall((source / "index.rst").read_text())
        for page in block.split()
    ]

    assert len(listed) >= 8, listed
    assert not [page for page in listed if not (source / f"{page}.rst").exists()]
