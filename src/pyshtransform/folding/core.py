"""Core functions for the folding transformation."""

import numpy as np


def fold_clm_numpy(
    unfolded_f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    c: np.ndarray[tuple[int], np.dtype[np.int_]],
    l: np.ndarray[tuple[int], np.dtype[np.int_]],
    m: np.ndarray[tuple[int], np.dtype[np.int_]],
    f: np.ndarray[tuple[int], np.dtype[np.float64]],
    clm: np.ndarray[tuple[int], np.dtype[np.int_]],
    *,
    folded_truncation: int,
    dtype: str,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Folds spectral coefficients.

    Args:
        unfolded_f_clm: Unfolded spectral coefficients.
        c: Indices for the `c` dimension in folded spectral space.
        l: Indices for the `l` dimension in folded spectral space.
        m: Indices for the `m` dimension in folded spectral space.
        f: Correction factors.
        clm: One-dimensional indices of the unfolded spectral coefficients.
        folded_truncation: Output truncation.
        dtype: Output floating-point data type.

    Returns:
        Folded spectral coefficients.
    """
    shape = list(unfolded_f_clm.shape)
    batch_shape = shape[:-1]
    folded_shape = [2, folded_truncation + 1, folded_truncation + 1]
    total_shape = tuple(batch_shape + folded_shape)
    folded_f_clm = np.zeros(total_shape, dtype=dtype)
    folded_f_clm[..., c, l, m] = f * unfolded_f_clm[..., clm]
    return folded_f_clm


def unfold_clm_numpy(
    folded_f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    c: np.ndarray[tuple[int], np.dtype[np.int_]],
    l: np.ndarray[tuple[int], np.dtype[np.int_]],
    m: np.ndarray[tuple[int], np.dtype[np.int_]],
    f: np.ndarray[tuple[int], np.dtype[np.float64]],
    clm: np.ndarray[tuple[int], np.dtype[np.int_]],
    *,
    unfolded_truncation: int,
    dtype: str,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Unfolds spectral coefficients.

    Args:
        folded_f_clm: Folded spectral coefficients.
        c: Indices for the `c` dimension in folded spectral space.
        l: Indices for the `l` dimension in folded spectral space.
        m: Indices for the `m` dimension in folded spectral space.
        f: Correction factors.
        clm: One-dimensional indices of the unfolded spectral coefficients.
        unfolded_truncation: Output truncation.
        dtype: Output floating-point data type.

    Returns:
        Unfolded spectral coefficients.
    """
    shape = list(folded_f_clm.shape)
    batch_shape = shape[:-3]
    unfolded_shape = [(unfolded_truncation + 1) * (unfolded_truncation + 2)]
    total_shape = tuple(batch_shape + unfolded_shape)
    unfolded_f_clm = np.zeros(total_shape, dtype=dtype)
    unfolded_f_clm[..., clm] = folded_f_clm[..., c, l, m] / f
    return unfolded_f_clm
