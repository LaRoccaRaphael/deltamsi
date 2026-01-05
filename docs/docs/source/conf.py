# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../../../src"))

project = 'Pymsix'
copyright = '2026, Raphaël La Rocca, Lionel La Rocca'
author = 'Raphaël La Rocca, Lionel La Rocca'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx_copybutton',
]
autosummary_generate = True
autodoc_typehints = "description"
templates_path = ['_templates']
exclude_patterns = []
autodoc_mock_imports = [
    "msicube",
    "numpy",
    "pandas",
    "anndata",
    "h5py",
    "sklearn",
    "scipy",
    "matplotlib",
    "seaborn"
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'imported-members': False, # TRÈS IMPORTANT : mettez à False
}

nitpick_ignore = [
    ('py:class', 'typing.Any'),
    ('py:class', 'Any'),
    ('py:class', 'pymsix.params.options.MeanSpectrumOptions'),
]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

