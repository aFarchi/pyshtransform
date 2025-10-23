#!/usr/bin/env python

import logging
import math

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class FoldingTransformation:
    def __init__(self, dtype, truncation, factor):
        self.dtype = dtype
        self.truncation = truncation
        self.factor = factor
        self.input_truncation = None
        self.folding_coefficients = None

    def precompute_folding_coefficients(self, input_truncation):
        # check if the coefficients have already been pre-computed
        if self.input_truncation == input_truncation:
            return
        # use input truncation if internal truncation is unspecified
        self.truncation = self.truncation or input_truncation
        # compute the folding coefficients
        self.folding_coefficients = precompute_folding_coefficients(
            input_truncation,
            self.truncation,
            self.dtype,
            self.factor,
        )
        # save input truncation
        self.input_truncation = input_truncation

    def enforce_dtype(self, ds_data):
        logger.info(f'enforcing dtype "{self.dtype}" before transformation')
        return ds_data.astype(self.dtype)

    def fold_clm(self, ds_data):
        logger.info('applying "fold_clm" transformation')
        num_clm = len(ds_data.clm)
        input_truncation = int((math.sqrt(4 * num_clm + 1) - 1) / 2) - 1
        self.precompute_folding_coefficients(input_truncation)
        ds_data = self.enforce_dtype(ds_data)
        return xr.apply_ufunc(
            fold_clm_numpy,
            ds_data,
            self.folding_coefficients['c'],
            self.folding_coefficients['l'],
            self.folding_coefficients['m'],
            self.folding_coefficients['f'],
            self.folding_coefficients['clm'],
            kwargs=dict(
                truncation=self.truncation,
                dtype=self.dtype,
            ),
            input_core_dims=[
                ['clm'],
                ['clm_truncated'],
                ['clm_truncated'],
                ['clm_truncated'],
                ['clm_truncated'],
                ['clm_truncated'],
            ],
            output_core_dims=[['c', 'l', 'm']],
            dask='parallelized',
            output_dtypes=[self.dtype],
            dask_gufunc_kwargs=dict(
                output_sizes=dict(
                    c=2,
                    l=self.truncation + 1,
                    m=self.truncation + 1,
                )
            ),
        )

    def unfold_clm(self, ds_data):
        logger.info('applying "unfold_clm" transformation')
        self.precompute_folding_coefficients(self.truncation)
        ds_data = self.enforce_dtype(ds_data)
        return xr.apply_ufunc(
            unfold_clm_numpy,
            ds_data,
            self.folding_coefficients['c'],
            self.folding_coefficients['l'],
            self.folding_coefficients['m'],
            self.folding_coefficients['f'],
            self.folding_coefficients['clm'],
            kwargs=dict(
                truncation=self.truncation,
                dtype=self.dtype,
            ),
            input_core_dims=[
                ['c', 'l', 'm'],
                ['clm'],
                ['clm'],
                ['clm'],
                ['clm'],
                ['clm'],
            ],
            output_core_dims=[['clm']],
            dask='parallelized',
            output_dtypes=[self.dtype],
        )


def precompute_folding_coefficients(input_truncation, target_truncation, dtype, factor):
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
    # returns all the coefficients in a dictionary
    return dict(
        clm=np.array(indices_clm),
        c=np.array(indices_c),
        l=np.array(indices_l),
        m=np.array(indices_m),
        f=np.array(factors, dtype=dtype),
    )


def fold_clm_numpy(unfolded_f_clm, c, l, m, f, clm, *, truncation, dtype):
    shape = list(unfolded_f_clm.shape)
    batch_shape = shape[:-1]
    folded_shape = [2] + [truncation + 1] * 2
    total_shape = tuple(batch_shape + folded_shape)
    folded_f_clm = np.zeros(total_shape, dtype=dtype)
    folded_f_clm[..., c, l, m] = f * unfolded_f_clm[..., clm]
    return folded_f_clm


def unfold_clm_numpy(folded_f_clm, c, l, m, f, clm, *, truncation, dtype):
    shape = list(folded_f_clm.shape)
    batch_shape = shape[:-3]
    unfolded_shape = [(truncation + 1) * (truncation + 2)]
    total_shape = tuple(batch_shape + unfolded_shape)
    unfolded_f_clm = np.zeros(total_shape, dtype=dtype)
    unfolded_f_clm[..., clm] = folded_f_clm[..., c, l, m] / f
    return unfolded_f_clm
