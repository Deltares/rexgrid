"""
This module contains the logic for regridding from a structured form to another
structured form. All coordinates are assumed to be fully orthogonal to each
other.

While the unstructured logic would work for structured data as well, it is much
less efficient than utilizing the structure of the coordinates.
"""
import numpy as np

from .utils import broadcast
from .overlap_1d import overlap_1d, overlap_1d_nd


class StructuredGrid1d:
    """
    e.g. z -> z; so also works for unstructured

    Parameters
    ----------
    bounds: (n, 2)
    """

    def __init__(self, bounds):
        self.bounds = bounds
        
    @property
    def size(self):
        return len(self.bounds)

    def overlap(self, other, relative: bool):
        source_index, target_index, weights = overlap_1d(self.bounds, other.bounds)
        if relative:
            weights /= self.length()[source_index]
        return source_index, target_index, weights

    def length(self):
        return abs(np.diff(self.bounds, axis=1))
    


class StructuredGrid2d:
    """
    e.g. (x,y) -> (x,y)

    Parameters
    ----------
    xbounds: (nx, 2)
    ybounds: (ny, 2)
    """

    def __init__(
        self,
        xbounds,
        ybounds,
    ):
        self.xbounds = StructuredGrid1d(xbounds)
        self.ybounds = StructuredGrid1d(ybounds)
        
    @property
    def shape(self):
        return (self.ybounds.size, self.xbounds.size)

    def overlap(self, other, relative: bool):
        source_index_x, target_index_x, weights_x = self.xbounds.overlap(other.xbounds, relative)
        source_index_y, target_index_y, weights_y = self.ybounds.overlap(other.ybounds, relative)
        return broadcast(
            self.shape,
            other.shape,
            (source_index_y, source_index_x),
            (target_index_y, target_index_x),
            (weights_y, weights_x),
        )


class StructuredGrid3d:
    """e.g. (x,y,z) -> (x,y,z)"""

    def __init__(self, xbounds, ybounds, zbounds):
        self.xbounds = StructuredGrid1d(xbounds)
        self.ybounds = StructuredGrid1d(ybounds)
        self.zbounds = StructuredGrid1d(zbounds)

    def overlap(self, other, relative):
        source_index_x, target_index_x, weights_x = self.xbounds.overlap(
            other.xbounds, relative
        )
        source_index_y, target_index_y, weights_y = self.ybounds.overlap(
            other.ybounds, relative
        )
        source_index_z, target_index_z, weights_z = self.zbounds.overlap(
            other.zbounds, relative
        )
        return broadcast(
            self.shape,
            other.shape,
            (source_index_z, source_index_y, source_index_x),
            (target_index_z, target_index_y, target_index_x),
            (weights_z, weights_y, weights_x),
        )
        

class ExplicitGrid:
    def __init__(self, zbounds):
        nz = zbounds.shape[-1]
        self.zbounds = zbounds.reshape(-1, nz, 2)

    def length(self):
        return np.diff(self.zbounds, axis=-1)
    
    def overlap(self, other, relative: bool):
        source_index = target_index = np.arange(self.zbounds.shape[0])
        source_index, target_index, weights = overlap_1d_nd(
            self.zbounds, other.zbounds, source_index, target_index
        )
        if relative:
            weights_z /= self.length.ravel()[source_index]
        return source_index, target_index, weights


class ExplicitStructuredGrid3d(ExplicitGrid):
    """
    e.g. z(y, x) -> z(y, x)
    
    Promote to explicit bounds if not yet available.
    """
    def __init__(self, xbounds, ybounds, zbounds):
        self.xbounds = StructuredGrid1d(xbounds)
        self.ybounds = StructuredGrid1d(ybounds)
        nz = zbounds.shape[-1]
        self.zbounds = zbounds.reshape(-1, nz, 2)
        
    def length(self):
        return np.diff(self.zbounds, axis=-1)
        
    def overlap(self, other, relative):
        source_index_x, target_index_x, weights_x = self.xbounds.overlap(
            other.xbounds, relative
        )
        source_index_y, target_index_y, weights_y = self.ybounds.overlap(
            other.ybounds, relative
        )
        source_index_yx, target_index_yx, weights_yx = broadcast(
            self.shape[1:],
            other.shape[1:],
            (source_index_y, source_index_x),
            (target_index_y, target_index_x),
            (weights_y, weights_x),
        )
        source_index, target_index, weights_z = overlap_1d_nd(
            self.zbounds, other.zbounds, source_index_yx, target_index_yx
        )
        if relative:
            weights_z /= self.length.ravel()[source_index]
        weights = weights_z * weights_yx
        return source_index, target_index, weights


GRIDS = {
#   1D 2D 3D
    (1, 0, 0): StructuredGrid1d,
    (2, 0, 0): StructuredGrid2d,
    (3, 0, 0): StructuredGrid3d,
    (0, 1, 0): ExplicitGrid,
    (0, 0, 1): ExplicitGrid,
    (2, 0, 1): ExplicitStructuredGrid3d,
}
