import numpy as np


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
