# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Fintuna'
copyright = '2022, Cortecs'
author = 'Cortecs GmbH'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    # 'sphinx.ext.doctest',
    'sphinx_toolbox.sidebar_links',
    'sphinx_toolbox.github',
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]
github_username = 'markoarnauto'
github_repository = 'fintuna'

autosummary_generate = True
# autoapi_dirs = ['../../fintuna']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
