
# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.join("../.."))
import geocode

# -- Project information -----------------------------------------------------

project = 'GeoCode'
copyright = '2026, geo-kit'
author = 'geo-kit'
release = '0.0.1'

# -- General configuration ---------------------------------------------------

extensions = [
	'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon'
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

html_theme = 'agogo'
html_static_path = ['_static']

autoclass_content = 'class'
add_module_names = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('http://pandas-docs.github.io/pandas-docs-travis/', None)
}

# -- Latex output -------------------------------------------------------------
latex_documents = [
 ('index', 'manual.tex', u'UserManual', u'GeoCode', 'manual'),
]
