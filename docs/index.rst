Ultralyzer documentation
========================

.. container:: hero-kicker

   MATLAB-first retinal analysis workflow

.. container:: hero-lead

   Ultralyzer is a desktop workflow for reviewing, segmenting, editing, and quantifying retinal ultra-widefield images. This documentation follows the current GUI, with a MATLAB-first setup path and task-based guidance for day-to-day analysis.

.. container:: hero-links

   :doc:`Start with installation <doc_installation>`
   :doc:`Follow the workflow <doc_user_guide>`
   :doc:`Tour the interface <doc_gui_explained>`

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Production path
      :class-card: hero-callout

      The documented production path uses MATLAB as the metric computation backend for geometry-dependent measurements.

   .. grid-item-card:: Graceful degradation
      :class-card: hero-callout

      Segmentation, review, editing, and export remain available when geometry support is partial, although some geometry-dependent metrics may be skipped.

.. image:: ./_static/images/gui_overview.png
   :alt: Ultralyzer main window overview
   :class: hero-screenshot

Start here
----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Install and configure
      :link: doc_installation
      :link-type: doc
      :class-card: doc-card

      Set up the application, configure MATLAB geometry backend, and understand what happens when geometry support is incomplete.

   .. grid-item-card:: Tour the interface
      :link: doc_gui_explained
      :link-type: doc
      :class-card: doc-card

      Learn the current window layout: top navigation, menu bar, status bar controls, review sidebar, and edit mode.

   .. grid-item-card:: Follow the workflow
      :link: doc_user_guide
      :link-type: doc
      :class-card: doc-card

      Load data, review images, segment structures, choose metric ROI regions, calculate metrics, and export results.

   .. grid-item-card:: Check reference material
      :link: doc_appendix
      :link-type: doc
      :class-card: doc-card

      Find keyboard shortcuts, metric definitions, and supporting reference material without interrupting the main workflow guide.

Current workflow context
------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: What is new in the current GUI?
      :class-card: accent-card

      The current interface adds a structured menu bar, compact status-bar controls, a dedicated review versus edit sidebar flow, metric ROI selection, and a geometry readiness indicator for metrics.

   .. grid-item-card:: What should users understand first?
      :class-card: accent-card

      Start with installation and the interface overview. The workflow guide assumes you know where the current-image actions, overlay controls, and edit tools live in the new layout.

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   doc_installation
   doc_gui_explained
   doc_user_guide
   doc_appendix