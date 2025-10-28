import numpy as np

class FoldingCoefficients:

    def __init__(self, input_truncation, target_truncation, dtype, factor):
        # full set of coefficients first
        tiles = (input_truncation + 1) * (input_truncation + 2) // 2
        full_indices_c = np.tile(np.arange(2), tiles)
        full_indices_m, full_indices_l = np.triu_indices(input_truncation + 1)
        full_indices_l = np.repeat(full_indices_l, 2)
        full_indices_m = np.repeat(full_indices_m, 2)
        full_factors = np.tile(np.array([1, -1]), tiles).astype(dtype)

        # apply correction factor if needed
        full_factors[2 * (input_truncation + 1) :] *= factor

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
            if i_l <= target_truncation and i_m <= target_truncation:
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
