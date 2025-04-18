# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyTENAX'
copyright = '2025, Petr Vohnicky'
author = 'Petr Vohnicky'
release = '0.0.1'

import os
import sys
from pathlib import Path

# Add the repository's src directory to sys.path
repo_path = Path(__file__).resolve().parents[2]  # Adjust according to the structure
src_path = repo_path / "src"
sys.path.insert(0, str(src_path))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 'sphinx_rtd_theme',
              'nbsphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
