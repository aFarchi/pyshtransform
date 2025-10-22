#!/usr/bin/env python

import numpy as np
from scipy.interpolate import BSpline
import xarray as xr

from pyshtransform.legendre import pre_glq


def compute_weight(da_dim):
    _, w = pre_glq(-1, 1, len(da_dim))
    return xr.DataArray(
        w * w.size / w.sum(),
        coords=(da_dim,),
    )


def compute_wavelet_matrix(dtype, truncation, spline_order, num_splines):
    if spline_order is None or num_splines is None or num_splines < 2:
        return None

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
        the_nodes[:] = nodes[i:i + spline_order + 2]
        the_spline = BSpline.basis_element(the_nodes, extrapolate=False)
        yy = the_spline(xx)
        yy = np.nan_to_num(yy, copy=False, nan=0)
        wavelet_matrix[i + 1] = yy

    # check that the sum is correct
    if not np.allclose(wavelet_matrix.sum(axis=0), 2 * np.ones(truncation + 1)):
        raise Exception

    return xr.DataArray(
        wavelet_matrix,
        dims=('wavelet_band', 'l'),
    ).assign_coords(
        wavelet_band=np.arange(1+num_splines)
    )
