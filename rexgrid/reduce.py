"""
Contains common reduction methods.
"""
import numpy as np


def get_method(method, methods):
    if isinstance(method, str):
        try:
            _method = methods[method]
        except KeyError as e:
            raise ValueError(
                "Invalid regridding method. Available methods are: {}".format(
                    methods.keys()
                )
            ) from e
    elif callable(method):
        _method = method
    else:
        raise TypeError("method must be a string or callable")
    return _method


def mean(values, indices, weights):
    vsum = 0.0
    wsum = 0.0
    for i, w in zip(indices, weights):
        v = values[i]
        if np.isnan(v):
            continue
        vsum += w * v
        wsum += w
    if wsum == 0:
        return np.nan
    else:
        return vsum / wsum


def harmonic_mean(values, indices, weights):
    v_agg = 0.0
    w_sum = 0.0
    for i, w in zip(indices, weights):
        v = values[i]
        w = weights[i]
        if np.isnan(v) or v == 0:
            continue
        if w > 0:
            w_sum += w
            v_agg += w / v
    if v_agg == 0 or w_sum == 0:
        return np.nan
    else:
        return w_sum / v_agg


def geometric_mean(values, indices, weights):
    v_agg = 0.0
    w_sum = 0.0

    # Compute sum to normalize weights to avoid tiny or huge values in exp
    normsum = 0.0
    for i, w in zip(indices, weights):
        normsum += w
    # Early return if no values
    if normsum == 0:
        return np.nan

    m = 0
    for i, w in zip(indices, weights):
        v = values[i]
        if np.isnan(v):
            continue
        if w > 0:
            v_agg += w * np.log(abs(v))
            w_sum += w
            if v < 0:
                m += 1

    if w_sum == 0:
        return np.nan
    else:
        return (-1.0) ** m * np.exp((1.0 / w_sum) * v_agg)


def sum(values, indices, weights):
    v_sum = 0.0
    w_sum = 0.0

    for i, w in zip(indices, weights):
        v = values[i]
        w = weights[i]
        if np.isnan(v):
            continue
        v_sum += v
        w_sum += w
    if w_sum == 0:
        return np.nan
    else:
        return v_sum


def minimum(values, indices, weights):
    vmin = values[indices[0]]
    for i in indices:
        v = values[i]
        if np.isnan(v):
            continue
        if v < vmin:
            vmin = v
    return vmin


def maximum(values, indices, weights):
    vmax = values[indices[0]]
    for i in indices:
        v = values[i]
        if np.isnan(v):
            continue
        if v > vmax:
            vmax = v
    return vmax


def mode(values, indices, weights):
    # Area weighted mode
    # Reuse weights to do counting: no allocations
    # The alternative is defining a separate frequency array in which to add
    # the weights. This implementation is less efficient in terms of looping.
    # With many unique values, it keeps having to loop through a big part of
    # the weights array... but it would do so with a separate frequency array
    # as well. There are somewhat more elements to traverse in this case.
    s = values.size
    w_sum = 0
    for i in range(s):
        v = values[i]
        w = weights[i]
        if np.isnan(v):
            continue
        w_sum += 1
        for j in range(i):  # Compare with previously found values
            if values[j] == v:  # matches previous value
                weights[j] += w  # increase previous weight
                break

    if w_sum == 0:  # It skipped everything: only nodata values
        return np.nan
    else:  # Find value with highest frequency
        w_max = 0
        for i in range(s):
            w = weights[i]
            if w > w_max:
                w_max = w
                v = values[i]
        return v


def median(values, indices, weights):
    return np.nanpercentile(values, 50)


def conductance(values, indices, weights):
    # Uses relative weights!
    # Rename to: first order conservative?
    v_agg = 0.0
    w_sum = 0.0
    for i, w in zip(indices, weights):
        v = values[i]
        if np.isnan(v):
            continue
        v_agg += v * w
        w_sum += w
    if w_sum == 0:
        return np.nan
    else:
        return v_agg


def max_overlap(values, weights):
    max_w = 0.0
    v = np.nan
    for i in range(values.size):
        w = weights[i]
        if w > max_w:
            max_w = w
            v = values[i]
    return v


OVERLAP_METHODS = {
    "nearest": "nearest",
    "mean": mean,
    "harmonic_mean": harmonic_mean,
    "geometric_mean": geometric_mean,
    "sum": sum,
    "minimum": minimum,
    "maximum": maximum,
    "mode": mode,
    "median": median,
    "conductance": conductance,
    "max_overlap": max_overlap,
}

METHODS = {
    "nearest": "nearest",
    "multilinear": "multilinear",
    "mean": mean,
    "harmonic_mean": harmonic_mean,
    "geometric_mean": geometric_mean,
    "sum": sum,
    "minimum": minimum,
    "maximum": maximum,
    "mode": mode,
    "median": median,
    "conductance": conductance,
    "max_overlap": max_overlap,
}
