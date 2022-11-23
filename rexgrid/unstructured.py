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
