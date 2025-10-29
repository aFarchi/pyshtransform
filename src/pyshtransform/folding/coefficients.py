"""Indices and factors for the folding transformation."""

import numpy as np


class FoldingCoefficients:
    """Folding coefficients.

    This class is essentially a container class. It contains
    the indices and factors which are used to fold / unfold
    arrays in spectral space.

    Attributes:
        clm: One-dimensional indices of the unfolded spectral coefficients.
        c: Indices for the `c` dimension in folded spectral space.
        l: Indices for the `l` dimension in folded spectral space.
        m: Indices for the `m` dimension in folded spectral space.
        f: Correction factors.
    """

    def __init__(
        self,
        unfolded_truncation: int,
        folded_truncation: int,
        dtype: str,
        factor: float,
    ):
        """Initialises the folding coefficients.

        Args:
            unfolded_truncation: Truncation in unfolded spectral space.
            folded_truncation: Truncation in folded spectral space.
            dtype: Floating-point data type.
            factor: Correction factor.
        """
        # full set of coefficients first
        tiles = (unfolded_truncation + 1) * (unfolded_truncation + 2) // 2
        full_indices_c = np.tile(np.arange(2), tiles)
        full_indices_m, full_indices_l = np.triu_indices(unfolded_truncation + 1)
        full_indices_l = np.repeat(full_indices_l, 2)
        full_indices_m = np.repeat(full_indices_m, 2)
        full_factors = np.tile(np.array([1, -1]), tiles).astype(dtype)

        # apply correction factor if needed
        full_factors[2 * (unfolded_truncation + 1) :] *= factor

        # drop coefficients beyond the internal truncation
        indices_clm = []
        indices_c = []
        indices_l = []
        indices_m = []
        factors = []
        for i_clm, (i_c, i_l, i_m, f) in enumerate(
            zip(
                full_indices_c,
                full_indices_l,
                full_indices_m,
                full_factors,
            )
        ):
            if i_l <= folded_truncation and i_m <= folded_truncation:
                indices_clm.append(i_clm)
                indices_c.append(i_c)
                indices_l.append(i_l)
                indices_m.append(i_m)
                factors.append(f)

        # save the coefficients as numpy arrays
        self.clm = np.array(indices_clm)
        self.c = np.array(indices_c)
        self.l = np.array(indices_l)
        self.m = np.array(indices_m)
        self.f = np.array(factors, dtype=dtype)
