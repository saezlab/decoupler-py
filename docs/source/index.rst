decoupler - Ensemble of methods to infer biological activities
==============================================================

**decoupler** is a package containing different statistical methods to extract biological activities from omics data within a unified framework. It allows to flexibly test any method with any prior knowledge resource and incorporates methods that take into account the sign and weight. It can be used with any omic, as long as its features can be linked to a biological process based on prior knowledge. For example, in transcriptomics gene sets regulated by a transcription factor, or in phospho-proteomics phosphosites that are targeted by a kinase.

This is its faster and memory efficient Python implementation, for the R version go `here <https://saezlab.github.io/decoupleR/>`_.

.. figure:: graphical_abstract.png
   :height: 500px
   :alt: decoupler’s workflow
   :align: center
   :class: no-scaled-link

   decoupler contains a collection of computational methods that coupled with 
   prior knowledge resources estimate biological activities from omics data.

Check out the `Usage <https://decoupler-py.readthedocs.io/en/latest/notebooks/usage.html>`_ or any other tutorial for further information.

License
-------
Footprint methods inside decoupler can be used for academic or commercial purposes, except ``viper`` which holds a non-commercial license. 

The data redistributed by OmniPath does not have a license, each original resource carries their own. 
`Here <https://omnipathdb.org/info>`_ one can find the license information of all the resources in OmniPath.

.. toctree::
   :maxdepth: 1
   :hidden:
   
   installation
   notebooks/usage
   notebooks/cell_annotation
   notebooks/progeny
   notebooks/dorothea
   notebooks/msigdb
   notebooks/pseudobulk
   api
   release_notes
   reference