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
]

autodoc_typehints = "description"
autodoc_inherit_docstrings = False
html_theme = "sphinx_rtd_theme"
html_title = "pyhighlights"
exclude_patterns = []
