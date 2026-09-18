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


def listed_pages(source: Path, page: str, seen: set) -> list:
    """Every page reachable from ``page`` through toctrees, depth first.

    The index names sections rather than pages, so a page is one or two
    toctrees away from it and a check that reads only the index would pass
    while a whole section was unreachable.
    """
    if page in seen:
        return []
    seen.add(page)
    path = source / f"{page}.rst"
    if not path.exists():
        return [page]
    found = [page]
    parent = Path(page).parent
    for block in TOCTREE.findall(path.read_text()):
        for entry in block.split():
            target = entry if entry.startswith("/") else str(parent / entry)
            found.extend(listed_pages(source, Path(target).as_posix(), seen))
    return found


def test_every_page_the_index_lists_is_there():
    """A toctree entry with no file behind it is a broken build."""
    source = ROOT / "docsrc" / "source"
    reachable = listed_pages(source, "index", set())

    assert len(reachable) >= 20, reachable
    assert not [page for page in reachable if not (source / f"{page}.rst").exists()]


def test_every_page_is_reachable_from_the_index():
    """A page no toctree names is a page the sidebar never offers."""
    source = ROOT / "docsrc" / "source"
    reachable = set(listed_pages(source, "index", set()))
    written = {
        path.relative_to(source).with_suffix("").as_posix()
        for path in source.rglob("*.rst")
    }

    assert not written - reachable, sorted(written - reachable)


def test_every_spp_model_has_an_api_page():
    """A merged architecture nobody can look up is half-delivered.

    The checks above resolve what the docs *name*, so a model the docs never
    mention passes them silently, which is how DR, MRD and DAR reached
    ``main`` with no page. This asks the question the other way round: every
    algorithm module under ``spp`` has to appear in an ``automodule``
    directive on some page.
    """
    #: Implemented but deliberately undocumented: grounded rationalization is
    #: still experimental, and the corpus it is written against is not public.
    #: Documenting it is what removes it from here.
    undocumented = {"grounded"}
    modules = {
        path.stem
        for path in (ROOT / "pyhighlights" / "components" / "models" / "spp").glob(
            "*.py"
        )
        if path.stem not in {"__init__", "base", "data", "implementations"}
    } - undocumented

    pages = (ROOT / "docsrc" / "source").rglob("*.rst")
    documented_modules = {
        module
        for page in pages
        for module in re.findall(
            r"automodule:: pyhighlights\.components\.models\.spp\.(\w+)",
            page.read_text(),
        )
    }

    assert modules, "no algorithm modules found; has the package moved?"
    assert not modules - documented_modules, sorted(modules - documented_modules)
