"""Implementation of the folding transformation in spectral space."""

import logging
import math

import xarray as xr

from pyshtransform.folding.coefficients import FoldingCoefficients
from pyshtransform.folding.core import fold_clm_numpy, unfold_clm_numpy

logger = logging.getLogger(__name__)


class FoldingTransformation:
    """Folding transformation in spectral space.

    Attributes:
        dtype: Floating-point data type.
        unfolded_truncation: Truncation in unfolded spectral space.
        folded_truncation: Truncation in folded spectral space.
        factor: Correction factor.
        folding_coefficients: Indices and factors for the transformation.
        variant: Transformation to apply in the `apply` method.
    """

    def __init__(
        self,
        dtype: str,
        unfolded_truncation: int,
        folded_truncation: int,
        factor: float,
        variant: str | None,
    ) -> None:
        """Initialises the folding transformation.

        Args:
            dtype: Output floating-point data type.
            unfolded_truncation: Truncation in unfolded spectral space.
            folded_truncation: Truncation in folded spectral space.
            factor: Correction factor.
            variant: Transformation to apply in the `apply` method.
        """
        self.dtype = dtype
        self.unfolded_truncation = unfolded_truncation
        self.folded_truncation = folded_truncation
        self.factor = factor
        self.folding_coefficients = FoldingCoefficients(
            unfolded_truncation=unfolded_truncation,
            folded_truncation=folded_truncation,
            dtype=dtype,
            factor=factor,
        )
        self.variant = variant

    def apply(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the selected transformation.

        Args:
            ds_data: Input dataset.

        Returns:
            Output dataset.

        Raises:
            ValueError: If the variant is unspecified.
        """
        if self.variant is None:
            message = 'please specify a variant'
            raise ValueError(message)
        ds_data_out: xr.Dataset = getattr(self, self.variant)(ds_data)
        return ds_data_out

    def enforce_dtype(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Enforces data type.

        Args:
            ds_data: Dataset.

        Returns:
            Dataset with the appropriate floating-point data type.
        """
        logger.info('enforcing dtype "%s" before transformation', self.dtype)
        return ds_data.astype(self.dtype)

    def fold_clm(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Transforms the dataset from unfolded to folded spectral space.

        Args:
            ds_data: Dataset containing unfolded spectral coefficients.

        Returns:
            Dataset containing folded spectral coefficients.

        Raises:
            ValueError: If the input dataset has incorrect truncation.
        """
        logger.info('applying "fold_clm" transformation')
        num_clm = len(ds_data.clm)
        unfolded_truncation = int((math.sqrt(4 * num_clm + 1) - 1) / 2) - 1
        if unfolded_truncation != self.unfolded_truncation:
            message = (
                f'unfolded truncation is incorrect,'
                f' expected {self.unfolded_truncation},'
                f' got {unfolded_truncation}'
            )
            raise ValueError(message)
        ds_data = self.enforce_dtype(ds_data)
        ds_data_out: xr.Dataset = xr.apply_ufunc(
            fold_clm_numpy,
            ds_data,
            self.folding_coefficients.c,
            self.folding_coefficients.l,
            self.folding_coefficients.m,
            self.folding_coefficients.f,
            self.folding_coefficients.clm,
            kwargs={
                'folded_truncation': self.folded_truncation,
                'dtype': self.dtype,
            },
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
            dask_gufunc_kwargs={
                'output_sizes': {
                    'c': 2,
                    'l': self.folded_truncation + 1,
                    'm': self.folded_truncation + 1,
                },
            },
        )
        return ds_data_out

    def unfold_clm(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Transforms the dataset from folded to unfolded spectral space.

        Args:
            ds_data: Dataset containing folded spectral coefficients.

        Returns:
            Dataset containing unfolded spectral coefficients.

        Raises:
            ValueError: If the input dataset has incorrect truncation.
        """
        logger.info('applying "unfold_clm" transformation')
        ds_data = self.enforce_dtype(ds_data)
        if (
            len(ds_data.c) != 2
            or len(ds_data.l) != self.folded_truncation + 1
            or len(ds_data.m) != self.folded_truncation + 1
        ):
            message = (
                f'folded shape is incorrect,'
                f' expected 2*{self.folded_truncation + 1}'
                f'*{self.folded_truncation + 1},'
                f' got {len(ds_data.c)}*{len(ds_data.l)}*{len(ds_data.m)}'
            )
            raise ValueError(message)
        ds_data_out: xr.Dataset = xr.apply_ufunc(
            unfold_clm_numpy,
            ds_data,
            self.folding_coefficients.c,
            self.folding_coefficients.l,
            self.folding_coefficients.m,
            self.folding_coefficients.f,
            self.folding_coefficients.clm,
            kwargs={
                'unfolded_truncation': self.unfolded_truncation,
                'dtype': self.dtype,
            },
            input_core_dims=[
                ['c', 'l', 'm'],
                ['clm_truncated'],
                ['clm_truncated'],
                ['clm_truncated'],
                ['clm_truncated'],
                ['clm_truncated'],
            ],
            output_core_dims=[['clm']],
            dask='parallelized',
            output_dtypes=[self.dtype],
            dask_gufunc_kwargs={
                'output_sizes': {
                    'clm': (self.unfolded_truncation + 1)
                    * (self.unfolded_truncation + 2),
                },
            },
        )
        return ds_data_out
