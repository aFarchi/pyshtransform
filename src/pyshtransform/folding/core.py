import numpy as np


def fold_clm_numpy(
    unfolded_f_clm: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    c: np.ndarray[tuple[int], np.dtype[np.int_]],
    l: np.ndarray[tuple[int], np.dtype[np.int_]],
    m: np.ndarray[tuple[int], np.dtype[np.int_]],
    f: np.ndarray[tuple[int], np.dtype[np.float64]],
    clm: np.ndarray[tuple[int], np.dtype[np.int_]],
    *,
    truncation: int,
    dtype: str,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    shape = list(unfolded_f_clm.shape)
    batch_shape = shape[:-1]
    folded_shape = [2] + [truncation + 1] * 2
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
    truncation: int,
    dtype: str,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    shape = list(folded_f_clm.shape)
    batch_shape = shape[:-3]
    unfolded_shape = [(truncation + 1) * (truncation + 2)]
    total_shape = tuple(batch_shape + unfolded_shape)
    unfolded_f_clm = np.zeros(total_shape, dtype=dtype)
    unfolded_f_clm[..., clm] = folded_f_clm[..., c, l, m] / f
    return unfolded_f_clm
