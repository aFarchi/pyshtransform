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
    return np.einsum(
        '...l,wl->...wl',
        f_clm,
        wavelet_matrix,
        casting='no',
        optimize=True,
    )


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

        - To compute the regular transformation, provide the Plm coefficients as
            `plm` and use `grad_phi=False`.
        - To compute the transformation with a longitude gradient, provide the Plm
            coefficients as `plm` and use `grad_phi=True`.
        - To compute the transformation with a latitude gradient, provide the Alm
            coefficients as `plm` and use `grad_phi=False`.

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
        '...im,iml->...lm', f_clm, pw, casting='no', optimize=True,
    )
    return f_grid


def grid_uv_to_folded_spec_vod_numpy(
    f_u,
    f_v,
    pw,
    alm,
    gauss_weights,
    cos_lat,
    *,
    radius=6_371_229.0,
):
    zonal_weights = pw / (radius * cos_lat[:, None, None])
    latitude_weights = 0.5 * gauss_weights[:, None, None] * alm / radius
    num_orders = pw.shape[-2]
    longitude_factor = -1j * np.arange(num_orders)

    u_fourier = np.fft.ihfft(f_u, axis=-1, norm='backward')[..., :num_orders]
    v_fourier = np.fft.ihfft(f_v, axis=-1, norm='backward')[..., :num_orders]

    def analyse(field, weights):
        return np.einsum('...im,iml->...lm', field, weights, optimize=True)

    vo = longitude_factor * analyse(v_fourier, zonal_weights)
    vo += analyse(u_fourier, latitude_weights)
    div = longitude_factor * analyse(u_fourier, zonal_weights)
    div -= analyse(v_fourier, latitude_weights)

    return (
        np.stack((vo.real, vo.imag), axis=-3),
        np.stack((div.real, div.imag), axis=-3),
    )


def folded_spec_vod_to_grid_uv_numpy(
    vo_clm,
    d_clm,
    plm,
    alm,
    cos_lat,
    *,
    num_lon,
    radius=6_371_229.0,
):
    degree = np.arange(plm.shape[-1])
    order = np.arange(plm.shape[-2])
    valid = degree[:, None] >= order[None, :]
    inverse_laplacian = np.zeros(degree.size, dtype=np.float64)
    inverse_laplacian[1:] = -radius**2 / (degree[1:] * (degree[1:] + 1))

    vo = vo_clm[..., 0, :, :] + 1j * vo_clm[..., 1, :, :]
    div = d_clm[..., 0, :, :] + 1j * d_clm[..., 1, :, :]
    psi = np.where(valid, vo, 0.0) * inverse_laplacian[:, None]
    chi = np.where(valid, div, 0.0) * inverse_laplacian[:, None]

    def synthesize(coefficients, basis):
        return np.einsum("...jm,imj->...im", coefficients, basis, optimize=True)

    longitude_factor = -1j * order
    zonal_scale = 1.0 / (radius * cos_lat[:, None])
    u_fourier = zonal_scale * synthesize(longitude_factor * chi, plm)
    u_fourier -= synthesize(psi, alm) / radius
    v_fourier = synthesize(chi, alm) / radius
    v_fourier += zonal_scale * synthesize(longitude_factor * psi, plm)

    return (
        np.fft.hfft(u_fourier, n=num_lon, axis=-1, norm="backward"),
        np.fft.hfft(v_fourier, n=num_lon, axis=-1, norm="backward"),
    )
