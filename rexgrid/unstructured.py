import numpy as np

from .structured import StructuredGrid1d
from .overlap_1d import overlap_1d_nd
from .utils import broadcast


class UnstructuredGrid2d:
    """
    e.g. face -> face

    Parameters
    ----------
    grid: Ugrid2d
    """

    def __init__(self, grid):
        self.grid = grid
        
    def shape(self):
        return self.grid.n_face

    def area(self):
        return self.grid.area

    def overlap(self, other, relative: bool):
        """
        Parameters
        ----------
        other: UnstructuredGrid2d
        """
        target_index, source_index, weights = self.grid.celltree.intersect_faces(
            vertices=other.grid.node_coordinates,
            faces=other.grid.face_node_connectivity,
            fill_value=other.grid.fill_value,
        )
        if relative:
            weights /= self.area[source_index]
        return source_index, target_index, weights


class UnstructuredPrismaticGrid3d(UnstructuredGrid2d):
    """
    e.g. (face, z) -> (face, z)

    Parameters
    ----------
    grid: Ugrid2d
    zbounds: (nlayer, 2)
    """

    def __init__(self, grid, zbounds):
        self.xygrid = UnstructuredGrid2d(grid)
        self.zbounds = StructuredGrid1d(zbounds)

    def overlap(self, other, relative):
        source_index_z, target_index_z, weights_z = self.zbounds.overlap(other.zbounds, relative)
        source_index_yx, target_index_yx, weights_yx = self.xygrid.overlap(other, relative)
        return broadcast(
            self.shape,
            other.shape,
            (source_index_z, source_index_yx),
            (target_index_z, target_index_yx),
            (weights_z, weights_yx),
        )


class ExplicitUnstructuredPrismaticGrid(UnstructuredGrid2d):
    """
    (face, z(face)) -> (face, z(face))

    Parameters
    ----------
    zbounds: (nlayer, nface, 2)
    """

    def __init__(self, grid, zbounds):
        self.xygrid = UnstructuredGrid2d(grid)
        self.zbounds = zbounds
        
    def length(self):
        return np.diff(self.zbounds, axis=-1)

    def overlap(self, other, relative):
        source_index_yx, target_index_yx, weights_yx = self.xygrid.overlap(other, relative)
        source_index, target_index, weights_z = overlap_1d_nd(
            self.zbounds, other.zbounds, source_index_yx, target_index_yx
        )
        if relative:
            weights_z /= self.length().ravel().sournce_index
        weights = weights_z * weights_yx
        return source_index, target_index, weights


GRIDS = {
#   1D,2D,3D
    (1, 0, 0): StructuredGrid1d,
    (0, 1, 0): UnstructuredGrid2d,
    (1, 1, 0): UnstructuredPrismaticGrid3d,
    (1, 0, 1): ExplicitUnstructuredPrismaticGrid,
}