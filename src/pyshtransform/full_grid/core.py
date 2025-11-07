"""Core functions for the spectral transformation with full Gaussian grids."""

import numpy as np


def apply_wavelet_decomposition_numpy(
    f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    wavelet_matrix: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Applies a wavelet decomposition to spectral coefficients.

    Args:
        f_clm: Folded spectral coefficients.
        wavelet_matrix: Wavelet matrix.

    Returns:
        Folded spectral coefficients decomposed into wavelets.
    """
    f_clm = np.einsum(
        '...l,wl->...wl',
        f_clm,
        wavelet_matrix,
        casting='no',
        optimize=True,
    )
    return f_clm


def apply_grad_phi_numpy(
    f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Computes the gradient with respect to longitude in spectral space.

    Args:
        f_clm: Folded spectral coefficients.

    Returns:
        Folded spectral coefficients representing the gradient of the
        input with respect to longitude.
    """
    shift = np.arange(f_clm.shape[-1])
    df_clm = np.zeros_like(f_clm)
    df_clm[..., 0, :, :] = shift * f_clm[..., 1, :, :]
    df_clm[..., 1, :, :] = -shift * f_clm[..., 0, :, :]
    return df_clm


def apply_mir_bug_numpy(
    f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Applies MIR bug in spectral space.

    Args:
        f_clm: Folded spectral coefficients.

    Returns:
        Folded spectral coefficients with MIR bug.
    """
    f_clm = f_clm.copy()
    f_clm[..., :, -1, -1] = 0
    return f_clm


def generic_folded_spec_to_grid_numpy(
    f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    plm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    *,
    num_lon: int,
    grad_phi: bool,
    mir_bug: bool,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Applies a generic transformation from folded spectral- to grid space.

    This function covers three cases:

    - to compute the "regular" transformation, provide the Plm coefficients as `plm` and use `grad_phi=False`;
    - to compute the transformation with gradient with respect to longitude, provide the Plm coefficients as `plm` and use `grad_phi=True`;
    - to compute the transformation with gradient with respect to latitude, provide the Alm coefficients as `plm` and use `grad_phi=False`.

    Args:
        f_clm: Folded spectral coefficients.
        plm: Coefficients used in Legendre transformation.
        num_lon: Number of longitude of the output.
        grad_phi: Whether to apply the gradient with respect to longitude.
        mir_bug: Whether to reproduce MIR bug.

    Returns:
        Grid space coefficients.
    """
    if mir_bug:
        f_clm = apply_mir_bug_numpy(f_clm)
    if grad_phi:
        f_clm = apply_grad_phi_numpy(f_clm)
    # apply Legendre transformation
    f = np.einsum('...jm,imj->...im', f_clm, plm, casting='no', optimize=True)
    # move to complex numbers
    f = f[..., 0, :, :] + 1j * f[..., 1, :, :]
    # apply hfft
    return np.fft.hfft(f, n=num_lon, axis=-1, norm='backward')


def grid_to_folded_spec_numpy(
    f: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    pw: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Applies the transformation from grid- to folded spectral space.

    Args:
        f: Grid space coefficients.
        pw: Weights of the Gauss--Legendre quadrature.

    Returns:
        Folded spectral coefficients.
    """
    # apply ihfft
    f_clm = np.fft.ihfft(f, axis=-1, norm='backward')
    # truncate the result
    f_clm = f_clm[..., : pw.shape[-1]]
    # move to real numbers
    f_clm = np.stack((f_clm.real, f_clm.imag), axis=-3)
    # apply inverse Legendre transformation
    f_grid: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.einsum(
        '...im,iml->...lm', f_clm, pw, casting='no', optimize=True
    )
    return f_grid
