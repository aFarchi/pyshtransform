"""Implementation of a wavelet matrix in spectral space."""

import numpy as np
import xarray as xr
from scipy.interpolate import BSpline


def compute_wavelet_matrix(
    dtype: str, truncation: int, spline_order: int | None, num_splines: int | None
) -> xr.DataArray:
    """Computes a wavelet matrix.

    Args:
        dtype: Floating-point data type.
        truncation: Truncation.
        spline_order: Order of the splines.
        num_splines: Number of splines.

    Returns:
        Data array containing the wavelet matrix.
    """
    if spline_order is None or num_splines is None or num_splines < 2:
        return xr.DataArray()

    # initialise wavelet matrix
    wavelet_matrix = np.zeros((1 + num_splines, truncation + 1), dtype=dtype)
    xx = 1 + np.arange(truncation + 1)

    # step 1: compute spline nodes
    power_min = 0
    power_max = np.log2(truncation + 1)
    num_internal_nodes = num_splines + 1 - spline_order
    delta = (power_max - power_min) / (num_internal_nodes - 1)
    left = delta * np.arange(-spline_order, 0) + power_min
    center = np.linspace(power_min, power_max, num_internal_nodes)
    right = delta * np.arange(1, spline_order + 1) + power_max
    log_nodes = np.concatenate((left, center, right))
    nodes = np.power(2, log_nodes)

    # step 2: apply splines
    wavelet_matrix[0] = 1
    for i in range(num_splines):
        the_nodes = np.zeros(spline_order + 2)
        the_nodes[:] = nodes[i : i + spline_order + 2]
        the_spline = BSpline.basis_element(the_nodes, extrapolate=False)
        yy = the_spline(xx)
        yy = np.nan_to_num(yy, copy=False, nan=0)
        wavelet_matrix[i + 1] = yy

    # check that the sum is correct
    if not np.allclose(wavelet_matrix.sum(axis=0), 2 * np.ones(truncation + 1)):
        raise ValueError('wavelet coefficients do not sum up to one')

    return xr.DataArray(
        wavelet_matrix,
        dims=('wavelet_band', 'l'),
    ).assign_coords(wavelet_band=np.arange(1 + num_splines))
