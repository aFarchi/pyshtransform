import logging
import math

import xarray as xr

from pyshtransform.folding.coefficients import FoldingCoefficients
from pyshtransform.folding.core import fold_clm_numpy, unfold_clm_numpy

logger = logging.getLogger(__name__)


class FoldingTransformation:
    def __init__(self, dtype: str, truncation: int, factor: float):
        self.dtype = dtype
        self.truncation = truncation
        self.factor = factor
        self.input_truncation: int | None = None
        self.folding_coefficients: FoldingCoefficients | None = None

    def precompute_folding_coefficients(self, input_truncation: int) -> None:
        # check if the coefficients have already been pre-computed
        if self.input_truncation == input_truncation:
            return
        # use input truncation if internal truncation is unspecified
        self.truncation = self.truncation or input_truncation
        # compute the folding coefficients
        self.folding_coefficients = FoldingCoefficients(
            input_truncation=input_truncation,
            target_truncation=self.truncation,
            dtype=self.dtype,
            factor=self.factor,
        )
        # save input truncation
        self.input_truncation = input_truncation

    def enforce_dtype(self, ds_data: xr.Dataset) -> xr.Dataset:
        logger.info(f'enforcing dtype "{self.dtype}" before transformation')
        return ds_data.astype(self.dtype)

    def fold_clm(self, ds_data: xr.Dataset) -> xr.Dataset:
        logger.info('applying "fold_clm" transformation')
        num_clm = len(ds_data.clm)
        input_truncation = int((math.sqrt(4 * num_clm + 1) - 1) / 2) - 1
        self.precompute_folding_coefficients(input_truncation)
        ds_data = self.enforce_dtype(ds_data)
        assert self.folding_coefficients is not None
        ds_data = xr.apply_ufunc(
            fold_clm_numpy,
            ds_data,
            self.folding_coefficients.c,
            self.folding_coefficients.l,
            self.folding_coefficients.m,
            self.folding_coefficients.f,
            self.folding_coefficients.clm,
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
        return ds_data

    def unfold_clm(self, ds_data: xr.Dataset) -> xr.Dataset:
        logger.info('applying "unfold_clm" transformation')
        self.precompute_folding_coefficients(self.truncation)
        ds_data = self.enforce_dtype(ds_data)
        assert self.folding_coefficients is not None
        ds_data = xr.apply_ufunc(
            unfold_clm_numpy,
            ds_data,
            self.folding_coefficients.c,
            self.folding_coefficients.l,
            self.folding_coefficients.m,
            self.folding_coefficients.f,
            self.folding_coefficients.clm,
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
        return ds_data
