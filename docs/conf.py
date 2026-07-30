# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from planktos import __version__, __copyright__, __author__

# -- Project information -----------------------------------------------------

project = 'Planktos'
copyright = __copyright__
author = __author__

version = __version__
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxext.opengraph'
]

# napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_admonition_for_examples = True
napoleon_custom_sections = ("Attributes", "params_style")

# -- Open Graph / link preview metadata --------------------------------------
# Controls how links to these docs render as cards on LinkedIn, Slack, etc.
# Without this, such links appear as blank, untitled boxes.

ogp_site_url = 'https://planktos.readthedocs.io/en/latest/'
ogp_site_name = 'Planktos documentation'
ogp_type = 'website'
# Pull an og:description out of each page's own text rather than repeating a
# single site-wide blurb on every card.
ogp_enable_meta_description = True
ogp_description_length = 200
# Card image. Must be an absolute URL: crawlers do not resolve relative paths.
# planktos_card.png is the left portion of logo.png (cut at its whitespace
# gutter) pre-sized to the 1200x630 / 1.91:1 Open Graph card ratio, so that
# platforms which crop to fill have nothing left to crop. Do not point this at
# logo.png directly: at 3.03:1 it gets center-cropped and loses the wordmark.
ogp_image = ogp_site_url + '_static/planktos_card.png'
ogp_image_alt = 'Planktos: agent-based modeling of small organisms in fluid flow'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc' #'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']