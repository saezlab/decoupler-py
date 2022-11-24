decoupler - Ensemble of methods to infer biological activities
==============================================================

|MainBuild| |Issues| |PyPIDownloads| |Docs| |Codecov|

.. |MainBuild| image:: https://github.com/saezlab/decoupler-py/actions/workflows/main.yml/badge.svg
   :target: https://github.com/saezlab/decoupler-py/actions
   
.. |Issues| image:: https://img.shields.io/github/issues/saezlab/decoupler-py.svg
   :target: https://github.com/saezlab/decoupler-py/issues/

.. |PyPIDownloads| image:: https://pepy.tech/badge/decoupler
   :target: https://pepy.tech/project/decoupler
   
.. |Docs| image:: https://readthedocs.org/projects/decoupler-py/badge/?version=latest
   :target: https://decoupler-py.readthedocs.io/en/latest/?badge=latest

.. |Codecov| image:: https://codecov.io/gh/saezlab/decoupler-py/branch/main/graph/badge.svg?token=TM0P29KKN5
   :target: https://codecov.io/gh/saezlab/decoupler-py

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

If you have any question or problem do not hesitate to open an `issue <https://github.com/saezlab/decoupler-py/issues>`_.

scverse
-------
**decoupler** is part of the `scverse <https://scverse.org>`_ ecosystem, a collection of tools for single-cell omics data analysis in python.
For more information check the link.

License
-------
Footprint methods inside decoupler can be used for academic or commercial purposes, except ``viper`` which holds a non-commercial license. 

The data redistributed by OmniPath does not have a license, each original resource carries their own. 
`Here <https://omnipathdb.org/info>`_ one can find the license information of all the resources in OmniPath.

Citation
-------
Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D., Müller-Dott S., Taus P., Dugourd A., Holland C.H., 
Ramirez Flores R.O. and Saez-Rodriguez J. 2022. decoupleR: ensemble of computational methods to infer biological activities 
from omics data. Bioinformatics Advances. https://doi.org/10.1093/bioadv/vbac016

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Main

   installation
   api
   release_notes
   reference

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Vignettes

   notebooks/usage
   notebooks/cell_annotation
   notebooks/progeny
   notebooks/dorothea
   notebooks/msigdb
   notebooks/pseudobulk
   notebooks/spatial
   notebooks/bulk
   notebooks/benchmark
