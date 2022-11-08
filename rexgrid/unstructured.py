import numpy as np

from .utils import create_linear_index, create_weights, overlap_1d_nd


class UnstructuredGridWrapper2d:
    """
    e.g. face -> face

    Parameters
    ----------
    grid: Ugrid2d
    """

    def __init__(self, grid):
        self.grid = grid

    def area(self):
        return self.grid.area

    def overlap(self, other):
        """
        Parameters
        ----------
        other: UnstructuredGridWrapper2d
        """
        target_index, source_index, weights = self.grid.celltree.intersect_faces(
            vertices=other.grid.node_coordinates,
            faces=other.grid.face_node_connectivity,
            fill_value=other.grid.fill_value,
        )
        return target_index, source_index, weights

    def relative_overlap(self, other):
        target_index, source_index, weights = self.overlap(other)
        weights /= self.area[source_index]
        return source_index, target_index, weights


class UnstructuredPrismaticGridWrapper3d(UnstructuredGridWrapper2d):
    """
    e.g. (face, z) -> (face, z)

    Parameters
    ----------
    grid: Ugrid2d
    zbounds: (nlayer, 2)
    """

    def __init__(self, grid, zbounds):
        self.grid = grid
        self.zbounds = StructuredGridWrapper1d(zbounds)

    def _broadcast_overlap(
        self,
        other,
        source_index_z,
        source_index_yx,
        target_index_z,
        target_index_yx,
        weights_z,
        weights_yx,
    ):
        source_index = create_linear_index(
            (source_index_z, source_index_yx), (self.zbounds.size, self.grid.n_face)
        )
        target_index = create_linear_index(
            (target_index_z, target_index_yx), (other.zbounds.size, other.grid.n_face)
        )
        weights = create_weights((weights_z, weights_yx))
        return source_index, target_index, weights

    def overlap(self, other):
        target_index_z, source_index_z, weights_z = self.zbounds(other.zbounds)
        target_index_yx, source_index_yx, weights_yx = super(self).overlap(other)
        return self._broadcast_overlap(
            other,
            source_index_z,
            source_index_yx,
            target_index_z,
            target_index_yx,
            weights_z,
            weights_yx,
        )

    def relative_overlap(self, other):
        target_index_z, source_index_z, weights_z = self.zbounds.relative_overlap(
            other.zbounds
        )
        target_index_yx, source_index_yx, weights_yx = super(self).relative_overlap(
            other
        )
        return self._broadcast_overlap(
            other,
            source_index_z,
            source_index_yx,
            target_index_z,
            target_index_yx,
            weights_z,
            weights_yx,
        )


class ExplicitUnstructuredPrismaticGridWrapper(UnstructuredGridWrapper2d):
    """
    (face, z(face)) -> (face, z(face))

    Parameters
    ----------
    zbounds: (nlayer, nface, 2)
    """

    def __init__(self, grid, zbounds):
        self.grid = grid
        self.zbounds = zbounds

    def volume(self):
        height = abs(np.diff(self.zbounds, axis=-1))
        return height * self.area()

    def overlap(self, other):
        target_index_yx, source_index_yx, weights_yx = super(self).overlap(other)
        target_index, source_index, weights_z = overlap_1d_nd(
            self.zbounds, other.zbounds, source_index_yx, target_index_yx
        )
        weights = weights_z * weights_yx
        return target_index, source_index, weights

    def relative_overlap(self, other):
        target_index, source_index, weights = self.overlap(other)
        weights /= self.volume().ravel()[source_index]
        return source_index, target_index, weights
