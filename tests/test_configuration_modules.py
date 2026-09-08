"""Guards on how the configuration package is laid out.

Both checks here cover failures that only appear once the configuration
package is split across files, and that a passing test suite does not
otherwise reveal: cinnamon reads registration modules statically and executes
them in filesystem order.
"""

import ast
from pathlib import Path

from cinnamon.registry import Registry

import pyhighlights

CONFIGURATIONS = Path(pyhighlights.__file__).parent / "configurations"


def modules() -> dict[str, ast.Module]:
    return {
        path.stem: ast.parse(path.read_text())
        for path in sorted(CONFIGURATIONS.glob("*.py"))
        if path.stem != "__init__"
    }


def registers(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == "register_method"
        for node in ast.walk(tree)
    )


def test_every_registering_module_binds_the_namespace_literal():
    """``NamespaceExtractor`` resolves ``namespace=NAMESPACE`` against literals
    bound in the same file, so importing the name is not enough: the package
    would advertise no namespace and become unusable as an external directory.
    """
    for name, tree in modules().items():
        if not registers(tree):
            continue
        literals = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "NAMESPACE"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        ]
        assert literals, f"{name}.py registers without a NAMESPACE literal"

    namespaces, mapping = Registry.parse_configuration_files(
        directories=[Path(pyhighlights.__file__).parent]
    )
    assert namespaces == ["pyhighlights"]
    assert set(mapping) == {"pyhighlights"}


def test_no_registering_module_imports_another_one():
    """Cinnamon executes each configuration file on its own and resolves the
    keys that appeared against *that* file's namespace. Importing a registering
    sibling fires its registrations under the importer, which then fails with a
    ``KeyError`` whenever the filesystem hands the importer over first.
    """
    trees = modules()
    registering = {name for name, tree in trees.items() if registers(tree)}

    for name, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            prefix, _, imported = node.module.rpartition(".")
            if prefix != "pyhighlights.configurations":
                continue
            assert imported not in registering, (
                f"{name}.py imports registering module {imported}.py"
            )
