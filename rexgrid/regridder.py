"""
Types are your friend!

Use xarray broadcasting for selection?

1. structured -> structured
2. structured -> unstructured
3. unstructured -> structured
4. unstructured -> unstructured

Note: i -> j can be inverted to j -> v.

1a. Overlap: easy, compute and broadcast-multiply.
1b. Interpolation: easy, bilinear between four neighbors.

2a. Overlap: search unstructured bounding boxes on structured. More efficient than
conversion to unstructured, building celltree, then searching.
2b. Interpolation: easy, bilinear between four neighbors.

3a. Overlap: inverse of 2a.
3b. Interpolation: requires natural neighbor, IDW.

4a. Overlap: search celltree. Compute overlap.
4b. Interpolation: requires natural neighbor, IDW.

Conclusion: 2a, 3a could be optimized by using a grid search than celltree search.
Others require different classes?
Not all logic differs. Probably better to create Grid classes which implement
weighting procedures?

*   *
  *

Xugrid explicitly targets spatial data.

While arbitrary dimensionality could be supported with sufficient (obtuse?)
abstraction, two to three spatial dimension is much easier to comprehend.

 or Adapter?

Based on dimension to regrid, decide on a regridder.
"""
import abc
from itertools import chain
from typing import Callable, NamedTuple, Optional, Tuple, Union

import numba
import numpy as np
import xarray as xr
import xugrid as xu

from xugrid import Ugrid2d, UgridDataArray
from . import reduce, weights_1d
from .typing import FloatArray, IntArray
from .unstructured import (
    ExplicitUnstructuredPrismaticGrid,
    UnstructuredGrid2d,
    UnstructuredPrismaticGrid3d,
)
from .weight_matrix import (
    create_weight_matrix,
    WeightMatrixCSR,
    nzrange,
)


def _check_ugrid(a, b):
    if a != b:
        raise ValueError("grid does not match source grid of regridder")


def _prepend(ds: xr.Dataset, prefix: str):
    vars = ds.data_vars
    dims = ds.dims
    name_dict = {v: f"{prefix}{v}" for v in chain(vars, dims)}
    return ds.rename(name_dict)


def _get_grid_variables(ds: xr.Dataset, prefix: str):
    ds = ds[[var for var in ds.data_vars if var.startswith(prefix)]]
    name_dict = {
        v: v.replace(prefix, "") if v.startswith(prefix) else v
        for v in chain(ds.data_vars, ds.dims)
    }
    return ds.rename(name_dict)


def fast_isel(object, indexer):
    """
    Might be smart?

    See: https://github.com/pydata/xarray/issues/2227
    """


class BaseRegridder(abc.ABC):
    def __init__(
        self,
        source: Ugrid2d,
        target: Ugrid2d,
        weights: Optional[Tuple] = None,
    ):
        self.source_grid = source
        self.target_grid = target
        if weights is None:
            self.compute_weights()
        else:
            self.source_index, self.target_index, self.weights = weights

    @abc.abstractmethod
    def compute_weights(self):
        """
        Compute the weights from source to target.
        """

    @abc.abstractmethod
    def regrid(self, object):
        """
        Create a new object by regridding.
        """

    @abc.abstractmethod
    def to_dataset(self):
        """
        Store the computed weights in a dataset for re-use.
        """

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset):
        """
        Reconstruct the regridder from a dataset with source, target indices
        and weights.
        """
        source = Ugrid2d.from_dataset(_get_grid_variables(dataset, "__source__"))
        target = Ugrid2d.from_dataset(_get_grid_variables(dataset, "__target__"))
        weights = (
            dataset["source_index"].values,
            dataset["target_index"].values,
            dataset["weights"].values,
        )
        return cls(
            source,
            target,
            weights,
        )


class UnstructuredNearestRegridder(BaseRegridder):
    def compute_weights(self):
        # TODO: dispatch on types?
        tree = self.source_grid.celltree
        self.source_index = tree.locate_points(self.target_grid.centroids)
        self.weights = xr.DataArray(
            data=np.where(self.source_index != -1, 1.0, np.nan),
            dims=[self.target_grid.face_dimension],
        )
        return

    def regrid(self, obj: UgridDataArray) -> UgridDataArray:
        """
        Regrid an object to the target grid topology.

        Parameters
        ----------
        obj: UgridDataArray

        Returns
        -------
        regridded: UgridDataArray
            The data regridded to the target grid. The target grid has been set
            as the face dimension.
        """
        grid = obj.ugrid.grid
        facedim = grid.face_dimension
        _check_ugrid(self.source_grid, grid)
        da = obj.obj.isel({facedim: self.source_index})
        da.name = obj.name
        uda = UgridDataArray(da, self.target_grid)
        uda = uda.rename({facedim: self.target_grid.face_dimension})
        uda = uda * self.weights.values
        return uda

    def to_dataset(self) -> xr.Dataset:
        source_ds = _prepend(self.source_grid.to_dataset(), "__source__")
        target_ds = _prepend(self.target_grid.to_dataset(), "__target__")
        regrid_ds = xr.Dataset(
            {
                "source_index": self.source_index,
                "target_index": np.nan,
                "weights": self.weights,
            },
        )
        return xr.merge((source_ds, target_ds, regrid_ds))


class UnstructuredOverlapRegridder:
    """
    Used for area or volume weighted means.
    """

    def __init__(
        self,
        source: UgridDataArray,
        target: UgridDataArray,
        method: Union[str, Callable] = "mean",
    ):
        self.source: Ugrid2d = UnstructuredGrid2d(source)
        self.target: Ugrid2d = UnstructuredGrid2d(target)
        func = reduce.get_method(method, reduce.OVERLAP_METHODS)
        self._setup_regrid(func)
        self.compute_weights()

    def _setup_regrid(self, func) -> None:
        """
        Use a closure to capture func.
        """

        f = numba.njit(func)

        @numba.njit(parallel=True)
        def _regrid(A: WeightMatrixCSR, source: FloatArray, out: FloatArray):
            for target_index in numba.prange(A.n):
                indices, weights = nzrange(A, target_index)
                out[target_index] = f(source, indices, weights)
            return

        self._regrid = _regrid
        return

    def compute_weights(self):
        self.source_index, self.target_index, self.weights = self.source.overlap(
            self.target
        )
        return

    @classmethod
    def from_dataset(self, dataset: xr.Dataset):
        return

    def regrid(self, object) -> UgridDataArray:
        grid = object.ugrid.grid
        facedim = grid.face_dimension
        # extradims = [dim for dim in object.dims if dim != facedim]
        # stacked = object.stack(extradims=extradims)

        A = create_weight_matrix(self.target_index, self.source_index, self.weights)
        target = self.target

        # make this delayed with dask.
        # optimize with isel/faster isel?
        # target = self.target_grid
        # results = []
        # for coords, object1d in stacked.groupby("extradims", create_index=False):
        #    source = object1d.values.ravel()
        #    out = np.full(target.n_face)
        #    self._regrid(A, source, out)
        #    da = xr.DataArray(
        #        out,
        #        coords=coords,
        #        dims=[target.face_dimension],
        #    )
        #    results.append(da)

        # regridded = xr.concat(results, dim="extradims").unstack("extradims")

        source = object.values.ravel()
        out = np.full(target.grid.n_face, np.nan)
        self._regrid(A, source, out)
        regridded = xr.DataArray(
            data=out,
            dims=[facedim],
            name=object.name,
        )

        return UgridDataArray(
            regridded,
            target.grid,
        )
        

def is_xarray(obj):
    return isinstance(obj, xr.DataArray)


def is_ugrid(obj):
    return isinstance(obj, xu.UgridDataArray)


def to_ascending(obj):
    """
    Ensure all coordinates are (monotonic) ascending.
    """
    