"""
This module contains the logic for regridding from a structured form to another
structured form. All coordinates are assumed to be fully orthogonal to each
other.

While the unstructured logic would work for structured data as well, it is much
less efficient than utilizing the structure of the coordinates.
"""
from typing import Union

import numba
import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

from .utils import alt_cumsum, create_linear_index, create_weights, overlap_1d


class StructuredGridWrapper1d:
    """
    e.g. z -> z; so also works for unstructured

    Parameters
    ----------
    bounds: (n, 2)
    """

    def __init__(self, bounds):
        self.bounds = bounds
        self.size = len(bounds)

    def overlap(self, other):
        return overlap_1d(self.bounds, other.bounds)

    def length(self):
        return abs(np.diff(self.bounds, axis=1))

    def relative_overlap(self, other):
        source_index, target_index, weights = self.overlap(other)
        weights /= self.length()[source_index]
        return source_index, target_index, weights


class StructuredGridWrapper2d:
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
        self.xbounds = StructuredGridWrapper1d(xbounds)
        self.ybounds = StructuredGridWrapper1d(ybounds)

    def _broadcast_overlap(
        self,
        other,
        source_index_y,
        source_index_x,
        target_index_y,
        target_index_x,
        weights_y,
        weights_x,
    ):
        source_index = create_linear_index(
            (source_index_y, source_index_x), (self.ybounds.size, self.xbounds.size)
        )
        target_index = create_linear_index(
            (target_index_y, target_index_x), (other.ybounds.size, other.xbounds.size)
        )
        weights = create_weights((weights_y, weights_x))
        return source_index, target_index, weights

    def overlap(self, other):
        source_index_x, target_index_x, weights_x = self.xbounds.overlap(other.xbounds)
        source_index_y, target_index_y, weights_y = self.ybounds.overlap(other.ybounds)
        return self._broadcast_overlap(
            other,
            source_index_y,
            source_index_x,
            target_index_y,
            target_index_x,
            weights_y,
            weights_x,
        )

    def relative_overlap(self, other):
        source_index_x, target_index_x, weights_x = self.xbounds.relative_overlap(
            other.xbounds
        )
        source_index_y, target_index_y, weights_y = self.ybounds.relative_overlap(
            other.ybounds
        )
        return self._broadcast_overlap(
            other,
            source_index_y,
            source_index_x,
            target_index_y,
            target_index_x,
            weights_y,
            weights_x,
        )


class StructuredGridWrapper3d:
    """e.g. (x,y,z) -> (x,y,z)"""

    def __init__(self, xbounds, ybounds, zbounds):
        self.xbounds = StructuredGridWrapper1d(xbounds)
        self.ybounds = StructuredGridWrapper1d(ybounds)
        self.zbounds = StructuredGridWrapper1d(zbounds)

    def _broadcast_overlap(
        self,
        other,
        source_index_z,
        source_index_y,
        source_index_x,
        target_index_z,
        target_index_y,
        target_index_x,
        weights_y,
        weights_x,
        weights_z,
    ):
        source_index = create_linear_index(
            (source_index_z, source_index_y, source_index_x),
            (self.ybounds.size, self.xbounds.size),
        )
        target_index = create_linear_index(
            (target_index_z, target_index_y, target_index_x),
            (other.ybounds.size, other.xbounds.size),
        )
        weights = create_weights((weights_y, weights_x))
        return source_index, target_index, weights

    def overlap(self, other):
        source_index_x, target_index_x, weights_x = self.xbounds.relative_overlap(
            other.xbounds
        )
        source_index_y, target_index_y, weights_y = self.ybounds.relative_overlap(
            other.ybounds
        )
        source_index_z, target_index_z, weights_z = self.zbounds.relative_overlap(
            other.zbounds
        )
        return self._broadcast_overlap(
            other,
            source_index_z,
            source_index_y,
            source_index_x,
            target_index_z,
            target_index_y,
            target_index_x,
            weights_y,
            weights_x,
            weights_z,
        )


class ExplicitStructuredGridWrapper2d:
    """e.g. z(x) -> z(x), also for unstructured"""


class ExplicitStructuredGridWrapper3d:
    """e.g. z(x,y) -> z(x, y)"""


def linear_interpolation_weights_1d(source_x, target_x):
    """
    Returns indices and weights for linear interpolation along a single dimension.
    A sentinel value of -1 is added for target cells that are fully out of bounds.

    Parameters
    ----------
    source_x : np.array
        vertex coordinates of source
    target_x: np.array
        vertex coordinates of target
    """
    # Cannot interpolate "between" only one point
    if not source_x.size > 2:
        raise ValueError(
            "source_x must larger than 2. Cannot interpolate with only a point"
        )

    xmin = source_x.min()
    xmax = source_x.max()

    # Compute midpoints for linear interpolation
    source_dx = np.diff(source_x)
    mid_source_x = source_x[:-1] + 0.5 * source_dx
    target_dx = np.diff(target_x)
    mid_target_x = target_x[:-1] + 0.5 * target_dx

    # From np.searchsorted docstring:
    # Find the indices into a sorted array a such that, if the corresponding
    # elements in v were inserted before the indices, the order of a would
    # be preserved.
    i = np.searchsorted(mid_source_x, mid_target_x) - 1
    # Out of bounds indices
    i[i < 0] = 0
    i[i > mid_source_x.size - 2] = mid_source_x.size - 2
    valid = (mid_target_x >= source_x[i]) & (mid_target_x < source_x[i + 1])
    i = i[valid]
    j = np.arange(mid_source_x.size)[valid]

    # -------------------------------------------------------------------------
    # Visual example: interpolate from source with 2 cells to target 3 cells
    # The period . marks the midpoint of the cell
    # The pipe | marks the cell edge
    #
    #    |_____._____|_____._____|
    #    source_x0      source_x1
    #
    #    |___.___|___.___|___.___|
    #        x0      x1      x2
    #
    # Then normalized weight for cell x1:
    # weight = (x1 - source_x0) / (source_x1 - source_x0)
    # -------------------------------------------------------------------------

    norm_weights = (mid_target_x - mid_source_x[i]) / (
        mid_source_x[i + 1] - mid_source_x[i]
    )
    # deal with out of bounds locations
    # we place a sentinel value of -1 here
    i[mid_target_x < xmin] = -1
    i[mid_target_x > xmax] = -1
    # In case it's just inside of bounds, use only the value at the boundary
    norm_weights[norm_weights < 0.0] = 0.0
    norm_weights[norm_weights > 1.0] = 1.0

    # Interlace i and weights
    jj = np.repeat(j, 2)
    ii = np.empty(jj.size, dtype=int)
    ii[::2] = i
    ii[1::2] = i + 1
    weights = np.empty(jj.size)
    weights[::2] = norm_weights
    weights[1::2] = 1.0 - norm_weights
    return i, j, weights
