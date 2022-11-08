"""
Types are your friend!

Use xarray broadcasting for selection?
"""
import abc
from itertools import chain
from typing import Callable, NamedTuple, Optional, Tuple, Union

import numba
import numpy as np
import xarray as xr
from numba.core.extending import register_jitable

from ..ugrid import Ugrid2d
from ..ugrid_dataset import UgridDataArray
from . import reduce, weights_1d
from .unstructured import (
    ExplicitUnstructuredPrismaticGridWrapper,
    UnstructuredGridWrapper2d,
    UnstructuredPrismaticGridWrapper3d,
)


IntArray = np.ndarray
FloatArray = np.ndarray


class WeightMatrixCSR(NamedTuple):
    """
    NamedTuple for easy ingestion by numba.
    """

    indptr: IntArray
    indices: IntArray
    weights: FloatArray
    n: int
    nnz: int


def create_weight_matrix(
    target_index: IntArray,
    source_index: IntArray,
    weights: FloatArray,
):
    i = np.cumsum(np.bincount(target_index))
    indptr = np.empty(i.size + 1)
    indptr[0] = 0
    indptr[1:] = i
    return WeightMatrixCSR(
        indptr,
        source_index,
        weights,
        indptr.size - 1,
        source_index.size,
    )


@numba.njit(inline="always")
def nzrange(A: WeightMatrixCSR, row: int) -> Tuple[IntArray, FloatArray]:
    """
    Return the indices and values of a single row
    """
    start = A.indptr[row]
    end = A.indptr[row + 1]
    return A.indices[start:end], A.weights[start:end]


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
        self.source: Ugrid2d = UnstructuredGridWrapper2d(source)
        self.target: Ugrid2d = UnstructuredGridWrapper2d(target)
        func = reduce.get_method(method, reduce.OVERLAP_METHODS)
        self._setup_regrid(func)
        self.compute_weights()

    def _setup_regrid(self, func) -> None:
        """
        Use a closure to capture func.
        """

        f = register_jitable(func)

        @numba.njit(parallel=True)
        def _regrid(A: WeightMatrixCSR, source: FloatArray, out: FloatArray):
            for target_index in numba.prange(A.n):
                indices, weights = nzrange(A, target_index)
                out[target_index] = f(source, indices, weights)
            return

        self._regrid = _regrid
        return

    def compute_weights(self):
        self.target_index, self.source_index, self.weights = self.source.overlap(
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


class UnstructuredRelativeOverlapRegridder(UnstructuredOverlapRegridder):
    def compute_weights(self):
        super().compute_weights()
        self.weights /= self.source.area()[self.source_index]


class StructuredGridWrapper:
    """
    Only supports axis-aligned coordinates (1D).
    """

    def __init__(self, coord_bounds):
        self.coord_bounds = coord_bounds

    def area(self, dim0, dim1):
        length0 = self.length(dim0)
        length1 = self.length(dim1)
        return np.multiply.outer(length0, length1)

    def volume(self, dim0, dim1, dim2):
        length0 = self.length(dim0)
        length1 = self.length(dim1)
        length2 = self.length(dim2)
        return np.multiply.outer(np.multiply.outer(length0, length1), length2)


class StructuredGridWrapper1d:
    """e.g. z -> z; also works for unstructured"""

    def overlap(self, other):
        return weights_1d.overlap_1d(self.xbounds, other.xbounds)


class StructuredGridWrapper2d:
    """e.g. (x,y) -> (x,y)"""

    def overlap(self, other):
        yw = weights_1d.overlap_1d(self.ybounds, other.ybounds)
        xw = weights_1d.overlap_1d(self.xbounds, other.xbounds)
        return np.multiply.outer(yw, xw)


class StructuredGridWrapper3d:
    """e.g. (x,y,z) -> (x,y,z)"""

    def overlap(self, other):
        yw = weights_1d.overlap_1d(self.ybounds, other.ybounds)
        xw = weights_1d.overlap_1d(self.xbounds, other.xbounds)
        zw = weights_1d.overlap_1d(self.zbounds, other.zbounds)
        return np.multiply.outer(
            zw,
            np.multiply.outer(yw, xw),
        )
