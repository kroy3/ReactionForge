"""Sphinx configuration for the ReactionForge documentation (Read the Docs)."""

import os
import sys

# Make `src` importable for the API reference
sys.path.insert(0, os.path.abspath('..'))

project = 'ReactionForge'
author = 'Kushal Raj Roy'
copyright = '2025, Kushal Raj Roy, University of Houston'

extensions = [
    'myst_parser',            # Markdown pages (the user guide is Markdown)
    'sphinx.ext.autodoc',     # API reference from docstrings
    'sphinx.ext.napoleon',    # Google-style "Args:/Returns:" docstrings
    'sphinx.ext.viewcode',    # "[source]" links in the API reference
]

source_suffix = {'.md': 'markdown'}
root_doc = 'index'
exclude_patterns = ['_build']

# GitHub-style heading anchors, so links like USER_GUIDE.md#71-train-... work
# both on GitHub and on Read the Docs.
myst_heading_anchors = 3

# The docs build doesn't install the heavy ML/chemistry stack; autodoc only
# needs these modules to exist, not to work.
autodoc_mock_imports = [
    'torch', 'torch_geometric', 'rdkit', 'numpy', 'pandas', 'sklearn',
    'scipy', 'matplotlib', 'seaborn', 'tqdm',
]
autodoc_member_order = 'bysource'
autodoc_default_options = {'members': True, 'show-inheritance': False}

html_theme = 'sphinx_rtd_theme'
html_title = 'ReactionForge documentation'
html_context = {
    'display_github': True,
    'github_user': 'kroy3',
    'github_repo': 'ReactionForge',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}
