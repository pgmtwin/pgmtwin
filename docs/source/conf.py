# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../pgmtwin"))
sys.path.insert(0, os.path.abspath("../../examples"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pgmtwin"
copyright = "2025, Lorenzo Fabris"
author = "Lorenzo Fabris"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google or NumPy style docstrings
    "sphinx_autodoc_typehints",  # optional, for nice type hint formatting
]

templates_path = ["_templates"]
exclude_patterns = []

autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "bizstyle"
html_static_path = ["_static"]
