Rexgrid
=======

** This is a work in progress. **

Rexgrid regrids grid data represented as `Xarray_` (structured) or `Xugrid_`
(unstructured) objects.

`Regridding_` is the process of converting gridded data from one grid to
another grid. Rexgrid does 2D and 3D regridding of structured gridded data,
represented as xarray objects, as well as (`layered_`) unstructured gridded
data, represented as xugrid objects.

It provides a number of regridding methods, based on area or volume overlap, as
well as interpolation routines.

It currently only supports Cartesian coordinates. See e.g. `xESMF_` instead for
regridding with a spherical Earth representation (note: EMSF is `not
available_` via conda-forge on Windows).

.. _Xarray: https://docs.xarray.dev/en/stable/index.html
.. _Xugrid: https://deltares.github.io/xugrid/
.. _Regridding: https://climatedataguide.ucar.edu/climate-tools/regridding-overview
.. _layered: https://ugrid-conventions.github.io/ugrid-conventions/#3d-layered-mesh-topology
.. _xEMSF: https://xesmf.readthedocs.io/en/latest/index.html
.. _not available: https://github.com/conda-forge/esmf-feedstock/issues/64
