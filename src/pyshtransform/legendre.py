"""Implementation of the Gauss--Legendre quadrature."""

import numpy as np
import xarray as xr


def legendre(
    n: int, z: np.ndarray[tuple[int], np.dtype[np.float64]]
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Computes the n-th Legendre polynomial and its first derivative.

    Args:
        n: Order of the Legendre polynomial.
        z: Nodes at which the polynomial should be applied.

    Returns:
        The value of the n-th Legendre polynomial at the given nodes
        and the associated derivatives.

    Notes:
        This function has been inspired by the implementation of
        the corresponding fortran function in SHTOOLS (https://shtools.github.io/SHTOOLS/).
    """
    p1 = np.ones(len(z))
    p2 = np.zeros(len(z))
    for j in range(1, n + 1):
        # j * P_j(z) = (2 * j - 1) * z * P_(j-1)(z) - (j - 1) * P_(j-2)(z)
        p3 = p2
        p2 = p1
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
    # (z^2 - 1)/n * P_n'(z) = z * P_n(z) - P_(n-1)(z)
    pp = n * (z * p1 - p2) / (z * z - 1)
    return p1, pp


def gauss_legendre_nodes(
    num_lat: int, max_iter: int = 1000, atol: float = 1e-15
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Computes the nodes and weights for the Gauss--Legendre quadrature.

    Args:
        num_lat: Number of latitude nodes.
        max_iter: Maximum number of iteration in Newton's root-finding algorithm.
        atol: Absolute tolerance in Newton's root-finding algorithm.

    Returns:
        The nodes for the Gauss--Legendre quadrature and the associated weights.
    """
    # initial guess for the first (num_lat+1)//2 zeros
    z = np.cos(np.pi * (1 + np.arange((num_lat + 1) // 2) - 0.25) / (num_lat + 0.5))
    # derivative
    pp = np.zeros(len(z))
    for _ in range(max_iter):
        # recursion
        z_previous = z
        # compute legendre polynomial and its derivative
        p, pp = legendre(num_lat, z)
        # newton's method
        z = z_previous - p / pp
        # stopping criterion
        if abs(z - z_previous).max() <= atol:
            break
    # corresponding weights
    w = 2 / ((1 - z * z) * pp * pp)
    # full set of zeros and weights
    if num_lat % 2 == 0:
        zeros = np.concat((z, -z[::-1]))
        weights = np.concat((w, w[::-1]))
    else:
        zeros = np.concat((z[:-1], -z[::-1]))
        weights = np.concat((w[:-1], w[::-1]))
    return zeros, weights


def gauss_legendre_weights(da_dim: xr.DataArray) -> xr.DataArray:
    """Computes the weights for the Gauss--Legendre quadrature.

    This function assumes that `da_dim` is a data array containing the
    nodes for the Gauss--Legendre quadrature and returns the associated
    weights. In practice, it does not use the values of `da_dim`, but only its
    length.

    Args:
        da_dim: Data array containing the nodes for the Gauss--Legendre quadrature.

    Returns:
        The weights for the Gauss--Legendre quadrature.
    """
    _, w = gauss_legendre_nodes(len(da_dim))
    return xr.DataArray(
        w * w.size / w.sum(),
        coords=(da_dim,),
    )


def plmbar_d1(
    lmax: int, z: np.ndarray[tuple[int], np.dtype[np.float64]]
) -> tuple[
    np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
]:
    """Computes the Plm and Alm coefficients for the Legendre transformation.

    Args:
        lmax: Maximum value of l.
        z: Gauss--Legendre nodes at which to compute the coefficients.

    Returns:
        The Plm and Alm coefficients.

    Notes:
        This function has been inspired by the implementation of
        the corresponding fortran function in SHTOOLS (https://shtools.github.io/SHTOOLS/).
    """
    p = np.zeros((len(z), lmax + 1, lmax + 1))
    dp1 = np.zeros((len(z), lmax + 1, lmax + 1))

    scalef = 1.0e-280

    sqr = np.sqrt(1 + np.arange(2 * lmax + 1))
    il, im = np.tril_indices(lmax + 1)
    f1_flat = sqr[2 * il] * sqr[2 * il - 2] / (sqr[il + im - 1] * sqr[il - im - 1])
    f2_flat = (
        sqr[2 * il]
        * sqr[il - im - 2]
        * sqr[il + im - 2]
        / (sqr[2 * il - 4] * sqr[il + im - 1] * sqr[il - im - 1])
    )
    f1 = np.zeros((lmax + 1, lmax + 1))
    f2 = np.zeros((lmax + 1, lmax + 1))
    f1[il, im] = f1_flat
    f2[il, im] = f2_flat

    u = np.sqrt((1.0 - z) * (1.0 + z))
    pm2 = np.ones(len(z))
    p[..., 0, 0] = 1
    dp1[..., 0, 0] = 0
    pm1 = sqr[2] * z
    p[..., 0, 1] = pm1
    dp1[..., 0, 1] = sqr[2]
    for l in range(2, lmax + 1):
        plm = f1[l, 0] * z * pm1 - f2[l, 0] * pm2
        p[..., 0, l] = plm
        dp1[..., 0, l] = l * (sqr[2 * l] / sqr[2 * l - 2] * pm1 - z * plm) / u**2
        pm2 = pm1
        pm1 = plm

    pmm = scalef
    rescalem = np.ones(len(z)) / scalef
    for m in range(1, lmax):
        rescalem = rescalem * u
        pmm = pmm * sqr[2 * m] / sqr[2 * m - 1]
        p[..., m, m] = pmm * rescalem
        dp1[..., m, m] = -m * z * p[..., m, m] / u**2
        pm2 = pmm
        pm1 = z * sqr[2 * m + 2] * pmm
        p[..., m, m + 1] = pm1 * rescalem
        dp1[..., m, m + 1] = (
            sqr[2 * m + 2] * p[..., m, m] - z * (m + 1) * p[..., m, m + 1]
        ) / u**2
        for l in range(m + 2, lmax + 1):
            plm = z * f1[l, m] * pm1 - f2[l, m] * pm2
            p[..., m, l] = plm * rescalem
            dp1[..., m, l] = (
                sqr[2 * l]
                * sqr[l - m - 1]
                * sqr[l + m - 1]
                / sqr[2 * l - 2]
                * p[..., m, l - 1]
                - z * l * p[..., m, l]
            ) / u**2
            pm2 = pm1
            pm1 = plm

    rescalem = rescalem * u
    pmm = pmm * sqr[2 * lmax] / sqr[2 * lmax - 1]
    p[..., lmax, lmax] = pmm * rescalem
    dp1[..., lmax, lmax] = -lmax * z * p[..., lmax, lmax] / u**2
    return p, dp1
