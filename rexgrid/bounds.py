from typing import Union

import numpy as np
import xarray as xr


def scalar_spacing(coords, spacing):
    dim = coords.dims[0]
    diff = coords.diff(dim)
    if not np.allclose(diff.values, spacing.item(), atol=abs(1.0e-4 * spacing.item())):
        raise ValueError(
            f"Spacing of {coords.name} does not match value of {spacing.name}"
        )
    if (diff < 0).any() and spacing > 0:
        raise ValueError(
            f"{coords.name} labels are descending while {spacing.name} is positive"
        )
    halfdiff = xr.full_like(coords, spacing * 0.5)
    return halfdiff


def array_spacing(coords, spacing):
    dim = coords.dims[0]
    diff = coords.diff(dim)
    if coords.size != spacing.size:
        raise ValueError(f"Size of {coords.name} does not match size of {spacing.name}")
    if (diff < 0).any() and (spacing > 0).any():
        raise ValueError(
            f"{coords.name} labels are descending while {spacing.name} is positive"
        )
    halfdiff = 0.5 * spacing
    return halfdiff


def implicit_spacing(coords):
    dim = coords.dims[0]
    if coords.size == 1:
        raise ValueError(
            f"Cannot derive spacing of 1-sized coordinate: {coords.name} \n"
            f"Set bounds yourself or assign a d{coords.name} variable with spacing"
        )
    halfdiff = 0.5 * coords.diff(dim).abs()
    return halfdiff


def coords_to_bounds(
    obj: Union[xr.DataArray, xr.Dataset],
    var: str,
    dim: None,
):
    if dim is None:
        dim = var

    boundsname = f"{var}_bounds"
    if boundsname in obj.coords:
        raise ValueError(f"Bounds variable name {boundsname} already exists")

    coords = obj[var]
    index = obj.indexes[var]
    if not (index.is_monotonic_increasing or index.is_monotonic_decreasing):
        raise ValueError(f"{var} is not monotonic")

    # e.g. rioxarray will set dx, dy as (scalar) values.
    spacing_name = f"d{var}"
    if spacing_name in obj.coords:
        spacing = obj[spacing_name]
        spacing_shape = spacing.shape
        if len(spacing_shape) > 1:
            raise NotImplementedError(
                f"More than one dimension in spacing variable: {spacing_name}"
            )

        if spacing_shape in ((), (1,)):
            halfdiff = scalar_spacing(coords, spacing)
        else:
            halfdiff = array_spacing(coords, spacing)
    # Implicit spacing
    else:
        halfdiff = implicit_spacing(coords)

    lower = coords - halfdiff
    upper = coords + halfdiff
    bounds = xr.concat([lower, upper], dim="bounds")
    return bounds


def bounds_to_vertices(
    bounds,
):
    vertices = np.concatenate((bounds[..., :, 0], bounds[..., -1:, 1]), axis=-1)
    return vertices
