# %%

import imod
import xarray as xr
import numpy as np

# %%

example = imod.util.empty_3d(
    100.0,
    0.0,
    10_000.0,
    -100.0,
    20_000.0,
    30_000.0,
    layer=[1, 2, 3, 4, 5]
)

# %%
thickness = example.copy(data=np.random.rand(5, 100, 100)) + 1.0
bottom = thickness.cumsum("layer")
top = bottom + thickness
test = example.assign_coords(top=top)

# %%
# top_bounds

interval_dtype = np.dtype([("start", float), ("end", float)])
zbounds = np.empty((5, 100, 100), dtype=interval_dtype)
topv = top.values
botv = bottom.values
zbounds["start"] = topv
zbounds["end"] = botv

darr = xr.DataArray(zbounds, coords=dict(thickness.coords), dims=["layer", "y", "x"])

# %%

test = example.assign_coords(z_bounds=darr)
indexes = list(test.indexes.keys())
coords = test.coords




# %%

zbounds = xr.DataArray(
    data=np.stack([topv, botv], axis=-1),
)

# %%

test2 = example.assign_coords(zbounds=zbounds)
# %%
original = xr.Dataset(
dict(
    variable=(
        ("ln_p", "latitude", "longitude"),
        np.arange(8, dtype="f4").reshape(2, 2, 2),
        {"ancillary_variables": "std_devs det_lim"},
    ),
    std_devs=(
        ("ln_p", "latitude", "longitude"),
        np.arange(0.1, 0.9, 0.1).reshape(2, 2, 2),
        {"standard_name": "standard_error"},
    ),
    det_lim=(
        (),
        0.1,
        {"standard_name": "detection_minimum"},
    ),
),
    dict(
        latitude=("latitude", [0, 1], {"units": "degrees_north"}),
        longitude=("longitude", [0, 1], {"units": "degrees_east"}),
        latlon=((), -1, {"grid_mapping_name": "latitude_longitude"}),
        latitude_bnds=(("latitude", "bnds2"), [[0, 1], [1, 2]]),
        longitude_bnds=(("longitude", "bnds2"), [[0, 1], [1, 2]]),
        areas=(
            ("latitude", "longitude"),
            [[1, 1], [1, 1]],
            {"units": "degree^2"},
        ),
        ln_p=(
            "ln_p",
            [1.0, 0.5],
            {
                "standard_name": "atmosphere_ln_pressure_coordinate",
                "computed_standard_name": "air_pressure",
            },
        ),
        P0=((), 1013.25, {"units": "hPa"}),
    ),
)
 
# %%

original["variable"].encoding.update(
    {"cell_measures": "area: areas", "grid_mapping": "latlon"},
)
original.coords["latitude"].encoding.update(
    dict(grid_mapping="latlon", bounds="latitude_bnds")
)
original.coords["longitude"].encoding.update(
    dict(grid_mapping="latlon", bounds="longitude_bnds")
)
original.coords["ln_p"].encoding.update({"formula_terms": "p0: P0 lev : ln_p"})
# %%

# %%
# Requires adding a bounds variable, which does not allow using DataArrays.
# Acceptable workaround for now? ...
