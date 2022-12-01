"""
Quick overview
==============

Here are a number of quick examples of how to get started with rexgrid. More
detailed explanation can be found in the rest of the documentation.

We'll start by importing a few essential packages.
"""
# %%

import xugrid as xu

import rexgrid

# %%

uds = xu.data.adh_san_diego()

# %%
# We can then grab one of the data variables as usual for xarray:

elev = uds["elevation"]
elev

# Plotting
# --------

elev.ugrid.plot(cmap="viridis")

# Regrid
# ------

regridder = rexgrid.OverlapRegridder(source=elev.ugrid.grid, target=elev.ugrid.grid)

# %%
