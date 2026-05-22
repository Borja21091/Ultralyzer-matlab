# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ultralyzer'
copyright = '2025, Borja Marin'
author = 'Borja Marin'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Auto-generate docs from code
    'sphinx.ext.napoleon',     # Support for Google-style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'myst_parser',             # <--- ENABLE MARKDOWN SUPPORT
    'sphinx_design',           # <--- CARDS / GRIDS / CALLOUTS
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'Ultralyzer Documentation'
html_css_files = ['css/custom.css']
html_theme_options = {
    'navigation_with_keys': True,
    'light_css_variables': {
        'color-foreground-primary': '#1b1740',
        'color-foreground-secondary': '#4d4a7f',
        'color-foreground-muted': '#7a74b1',
        'color-foreground-border': '#b7b2e6',
        'color-background-primary': '#fbf9ff',
        'color-background-secondary': '#f3efff',
        'color-background-hover': '#ece6ff',
        'color-background-border': '#dfd8ff',
        'color-brand-primary': '#16d9ff',
        'color-brand-content': '#ff33cc',
        'color-brand-visited': '#7f5cff',
        'color-card-border': '#dfd8ff',
        'color-card-background': '#ffffff',
        'color-card-marginals-background': '#f6f1ff',
        'color-sidebar-background': '#f5f0ff',
        'color-sidebar-background-border': '#ddd5ff',
        'color-sidebar-link-text': '#5e5a92',
        'color-sidebar-link-text--top-level': '#0cbfe6',
        'color-sidebar-item-background--current': 'rgba(255, 51, 204, 0.10)',
        'color-sidebar-search-background': '#ffffff',
        'color-sidebar-search-background--focus': '#ffffff',
        'color-sidebar-search-border': '#d8cdfa',
        'color-sidebar-search-icon': '#7a74b1',
        'color-toc-background': '#f8f5ff',
        'color-toc-title-text': '#7a74b1',
        'color-toc-item-text': '#5e5a92',
        'color-toc-item-text--hover': '#1b1740',
        'color-toc-item-text--active': '#0cbfe6',
        'color-inline-code-background': '#f1edff',
        'color-admonition-background': '#faf6ff',
    },
    'dark_css_variables': {
        'color-foreground-primary': '#eef0ff',
        'color-foreground-secondary': '#bfc5ff',
        'color-foreground-muted': '#7f87c6',
        'color-foreground-border': '#252f66',
        'color-background-primary': '#060816',
        'color-background-secondary': '#0a0d22',
        'color-background-hover': '#12163a',
        'color-background-hover--transparent': '#12163a00',
        'color-background-border': '#1c2455',
        'color-brand-primary': '#18e4ff',
        'color-brand-content': '#ff3bd4',
        'color-brand-visited': '#8e63ff',
        'color-card-border': '#18214d',
        'color-card-background': '#0a1027',
        'color-card-marginals-background': '#11173a',
        'color-sidebar-background': '#040714',
        'color-sidebar-background-border': '#151c45',
        'color-sidebar-brand-text': '#f7f6ff',
        'color-sidebar-caption-text': '#7f87c6',
        'color-sidebar-link-text': '#c7caff',
        'color-sidebar-link-text--top-level': '#29e7ff',
        'color-sidebar-item-background': 'transparent',
        'color-sidebar-item-background--current': 'rgba(255, 59, 212, 0.14)',
        'color-sidebar-item-expander-background--hover': '#171d43',
        'color-sidebar-search-text': '#eef0ff',
        'color-sidebar-search-background': '#0d1230',
        'color-sidebar-search-background--focus': '#13183f',
        'color-sidebar-search-border': '#242d66',
        'color-sidebar-search-icon': '#7f87c6',
        'color-toc-background': '#090c20',
        'color-toc-title-text': '#7f87c6',
        'color-toc-item-text': '#c7caff',
        'color-toc-item-text--hover': '#eef0ff',
        'color-toc-item-text--active': '#18e4ff',
        'color-content-foreground': '#eef0ff',
        'color-inline-code-background': '#12163a',
        'color-admonition-background': '#0c112b',
        'color-highlighted-background': '#231451',
    },
}

pygments_style = 'sphinx'
pygments_dark_style = 'native'

# Support both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

myst_heading_anchors = 3
