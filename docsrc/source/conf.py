import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "pyhighlights"
copyright = "2026, Federico Ruggeri"
author = "Federico Ruggeri"

extensions = [
    "sphinx.ext.autodoc",
    # cinnamon documents its API in Google style, and these pages render
    # cinnamon symbols imported into the configuration modules.
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    # Method diagrams are written as text beside the prose they explain, so a
    # figure is diffable and never drifts from the page it belongs to.
    "sphinxcontrib.mermaid",
    # Builds the cinnamon registry at documentation-build time and renders
    # every registration as a card, so the page cannot drift from the keys.
    "registry_cards",
]

autodoc_typehints = "description"
autodoc_inherit_docstrings = False
html_theme = "pydata_sphinx_theme"
html_title = "pyhighlights"
html_static_path = ["_static"]
# The theme reserves a narrower column than a labelled objective and a wide
# table need, so the width is widened here rather than per page.
html_css_files = ["width.css"]
html_theme_options = {
    "github_url": "https://github.com/nlp-unibo/pyhighlights",
    "use_edit_page_button": False,
    "show_prev_next": True,
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}
html_context = {"default_mode": "auto"}
# A page with no siblings has nothing to put in the section sidebar, so the
# sidebar is dropped there rather than rendered empty under its heading.
html_sidebars = {
    "index": [],
    "concepts/select-then-predict": [],
    "project/contributing": [],
}
exclude_patterns = []
