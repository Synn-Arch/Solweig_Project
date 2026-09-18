# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'SOLWEIG-GPU'
copyright = '2022-2026, Harsh Kamath and Naveen Sudharsan'
author = 'Harsh Kamath and Naveen Sudharsan'

# Try to get version from package, fallback to static version
try:
    from solweig_gpu import __version__
    release = __version__
except ImportError:
    release = '2.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',  # For Markdown support
    'nbsphinx',  # For Jupyter notebooks
]

# Napoleon settings (for NumPy/Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_mock_imports = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'gdal', 
    'osgeo',
    'osgeo.gdal',
    'osgeo.osr',
    'osgeo.ogr',
    '_gdal',
    'osr', 
    'ogr', 
    'netCDF4',
    'scipy',
    'scipy.ndimage',
    'scipy.spatial',
    'numpy',
    'pandas',
    'xarray',
    'shapely',
    'shapely.geometry',
    'timezonefinder',
    'pytz',
    'matplotlib',
    'matplotlib.path',
    'tqdm',  # Progress bar library
    # Optional dependencies used by create_inputs / wind_ext_coeff / calculate_wbgt
    'rasterio',
    'rasterio.features',
    'rasterio.merge',
    'rasterio.transform',
    'rasterio.warp',
    'ee',
    'geemap',
    'geopandas',
    'osmnx',
    'geopy',
    'geopy.geocoders',
    'numba',
    'pyproj',
    'requests',
]

# Additional autodoc settings to handle import errors
autodoc_inherit_docstrings = True
autodoc_class_signature = "separated"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Internal spec/orchestration and benchmark-evidence pages under
# docs/incremental_design_tool/ (agent/, universal_editing/, date-stamped
# benchmarks) are linked from other pages but intentionally kept out of the
# navigation toctrees. Do not warn about such out-of-toctree documents;
# links to them still resolve because the pages stay in the build.
suppress_warnings = ['toc.not_included']

# The example notebooks contain no Markdown heading cells, so nbsphinx would
# emit "Each notebook should have at least one section title" and the
# notebooks.rst toctree would warn that the documents have no title.
# Generate an explicit page title (matching the toctree captions) here
# instead of editing the .ipynb sources. Paths are derived from env.docname
# (e.g. "notebooks/Example_ERA5") because env.doc2path() emits a
# RemovedInSphinx10Warning for string paths, which -W treats as an error.
nbsphinx_prolog = """
{% set titles = {
    'notebooks/Example_ownmetfile': 'Example 1: Using Your Own Met File',
    'notebooks/Example_ERA5': 'Example 2: Using ERA5 Data',
    'notebooks/Example_wrfout': 'Example 3: Using WRF Output',
} %}
{% set title = titles.get(env.docname, env.docname) %}

{{ title }}
{{ '=' * (title|length) }}

.. note::
   This page was generated from a Jupyter notebook.
   You can download it here: :download:`../{{ env.docname }}.ipynb`
"""

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme
html_static_path = ['_static']

# Logo
html_logo = '_static/logo.jpg'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    # Uncomment to show logo without text:
    # 'logo_only': True,
    # 'style_nav_header_background': '#2980B9',  # Custom header color
}

# The master toctree document.
master_doc = 'index'

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# MyST parser settings (for Markdown)
# Auto-generate anchors for headings up to level 4 so links like
# configuration.md#base_path and developer_guide.md#pipeline-stages resolve.
myst_heading_anchors = 4

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# nbsphinx configuration
nbsphinx_execute = 'never'  # Don't execute notebooks during build (faster, safer)
nbsphinx_allow_errors = True  # Continue build even if notebook has errors

# nbsphinx highlights code cells with the notebook's pygments_lexer metadata
# ("ipython3" for these notebooks). IPython 9 ships that lexer via the
# separate ipython_pygments_lexers package, and pygments entry-point discovery
# does not always find it inside a Sphinx build. Register it explicitly so
# notebook code cells never fall back with "Pygments lexer name ... not known".
from sphinx.highlighting import lexers

try:
    from ipython_pygments_lexers import IPython3Lexer

    lexers['ipython3'] = IPython3Lexer()
except ImportError:  # pragma: no cover - optional dependency
    pass

# Imported design-doc sources (docs/incremental_design_tool/universal_editing/)
# use ```mermaid fences. Sphinx ships no Mermaid lexer; alias it to the
# plain-text lexer so those blocks render as source text instead of failing
# the -W build with "Pygments lexer name 'mermaid' is not known".
from pygments.lexers import TextLexer

lexers['mermaid'] = TextLexer()

