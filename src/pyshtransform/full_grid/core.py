import numpy as np


def apply_wavelet_decomposition_numpy(
    f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    wavelet_matrix: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    f_clm = np.einsum(
        '...l,wl->...wl',
        f_clm,
        wavelet_matrix,
        casting='no',
    )
    return f_clm


def apply_grad_phi_numpy(
    f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    shift = np.arange(f_clm.shape[-1])
    df_clm = np.zeros_like(f_clm)
    df_clm[..., 0, :, :] = shift * f_clm[..., 1, :, :]
    df_clm[..., 1, :, :] = -shift * f_clm[..., 0, :, :]
    return df_clm


def apply_mir_bug_numpy(
    f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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
    if mir_bug:
        f_clm = apply_mir_bug_numpy(f_clm)
    if grad_phi:
        f_clm = apply_grad_phi_numpy(f_clm)
    # apply Legendre transformation
    f = np.einsum('...jm,imj->...im', f_clm, plm, casting='no')
    # move to complex numbers
    f = f[..., 0, :, :] + 1j * f[..., 1, :, :]
    # apply hfft
    return np.fft.hfft(f, n=num_lon, axis=-1, norm='backward')


def grid_to_folded_spec_numpy(
    f: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    pw: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    # apply ihfft
    f_clm = np.fft.ihfft(f, axis=-1, norm='backward')
    # truncate the result
    f_clm = f_clm[..., : pw.shape[-1]]
    # move to real numbers
    f_clm = np.stack((f_clm.real, f_clm.imag), axis=-3)
    # apply inverse Legendre transformation
    f_grid: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.einsum(
        '...im,iml->...lm', f_clm, pw, casting='no'
    )
    return f_grid
