"""Implementation of the spectral transformations using full Gaussian grids."""

import logging

import xarray as xr

from pyshtransform.folding.transformation import FoldingTransformation
from pyshtransform.full_grid.core import (
    apply_wavelet_decomposition_numpy,
    generic_folded_spec_to_grid_numpy,
    grid_to_folded_spec_numpy,
)
from pyshtransform.full_grid.grid import FullGrid
from pyshtransform.wavelet import compute_wavelet_matrix

logger = logging.getLogger(__name__)


class FullGridSphericalHarmonicsTransform(FoldingTransformation):
    """Spectral transformation using full Gaussian grids.

    Attributes:
        num_lat: Number of latitude nodes.
        num_lon: Number of longitude nodes.
        grid: Full Gaussian grid.
        wavelet_matrix: Wavelet matrix.
    """

    def __init__(
        self,
        dtype: str,
        unfolded_truncation: int,
        folded_truncation: int,
        num_lat: int,
        num_lon: int,
        spline_order: int | None,
        num_splines: int | None,
        variant: str | None,
    ):
        """Initialises the spectral transformation.

        Args:
            dtype: Output floating-point data type.
            unfolded_truncation: Truncation in unfolded spectral space.
            folded_truncation: Truncation in folded spectral space.
            num_lat: Number of latitude nodes.
            num_lon: Number of longitude nodes.
            spline_order: Spline order for the wavelet matrix.
            num_splines: Number of splines for the wavelet matrix.
            variant: Transformation to apply in the `apply` method.
        """
        super().__init__(
            dtype=dtype,
            unfolded_truncation=unfolded_truncation,
            folded_truncation=folded_truncation,
            factor=1,
            variant=variant,
        )
        self.num_lat = num_lat
        self.num_lon = num_lon
        self.grid = FullGrid(
            dtype=dtype,
            truncation=folded_truncation,
            num_lat=num_lat,
            num_lon=num_lon,
        )
        self.wavelet_matrix = (
            None
            if spline_order is None or num_splines is None or num_splines < 2
            else compute_wavelet_matrix(
                dtype=dtype,
                truncation=folded_truncation,
                spline_order=spline_order,
                num_splines=num_splines,
            )
        )

    def apply_wavelet_decomposition(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the wavelet decomposition in spectral space.

        Args:
            ds_data: Dataset containing folded spectral coefficients.

        Returns:
            Dataset containing folded spectral coefficients decomposed into wavelets.
        """
        if self.wavelet_matrix is None:
            return ds_data
        logger.info('applying "wavelet_decomposition" transformation')
        ds_data = self.enforce_dtype(ds_data)
        ds_data = xr.apply_ufunc(
            apply_wavelet_decomposition_numpy,
            ds_data,
            self.wavelet_matrix,
            input_core_dims=[
                ['l'],
                ['wavelet_band', 'l'],
            ],
            output_core_dims=[['wavelet_band', 'l']],
            dask='parallelized',
            output_dtypes=[self.dtype],
        )
        return ds_data

    def folded_spec_to_grid(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the transformation from folded spectral- to grid space.

        Args:
            ds_data: Dataset containing folded spectral coefficients.

        Returns:
            Dataset containing grid coefficients.
        """
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid" transformation')
        ds_data = self.enforce_dtype(ds_data)
        ds_data = xr.apply_ufunc(
            generic_folded_spec_to_grid_numpy,
            ds_data,
            self.grid.plm,
            kwargs=dict(
                num_lon=self.num_lon,
                grad_phi=False,
                mir_bug=False,
            ),
            input_core_dims=[
                ['c', 'l', 'm'],
                ['latitude', 'l', 'm'],
            ],
            output_core_dims=[['latitude', 'longitude']],
            dask='parallelized',
            output_dtypes=[self.dtype],
            dask_gufunc_kwargs=dict(
                output_sizes=dict(
                    longitude=self.num_lon,
                )
            ),
        ).assign_coords(
            latitude=self.grid.lat,
            longitude=self.grid.lon,
        )
        return ds_data

    def unfolded_spec_to_grid(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the transformation from unfolded spectral- to grid space.

        Args:
            ds_data: Dataset containing unfolded spectral coefficients.

        Returns:
            Dataset containing grid coefficients.
        """
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid(ds_data)

    def folded_spec_to_grid_mir(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the transformation from folded spectral- to grid space with MIR bug.

        Args:
            ds_data: Dataset containing folded spectral coefficients.

        Returns:
            Dataset containing grid coefficients.
        """
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid_mir" transformation')
        ds_data = self.enforce_dtype(ds_data)
        ds_data = xr.apply_ufunc(
            generic_folded_spec_to_grid_numpy,
            ds_data,
            self.grid.plm,
            kwargs=dict(
                num_lon=self.num_lon,
                grad_phi=False,
                mir_bug=True,
            ),
            input_core_dims=[
                ['c', 'l', 'm'],
                ['latitude', 'l', 'm'],
            ],
            output_core_dims=[['latitude', 'longitude']],
            dask='parallelized',
            output_dtypes=[self.dtype],
            dask_gufunc_kwargs=dict(
                output_sizes=dict(
                    longitude=self.num_lon,
                )
            ),
        ).assign_coords(
            latitude=self.grid.lat,
            longitude=self.grid.lon,
        )
        return ds_data

    def unfolded_spec_to_grid_mir(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the transformation from unfolded spectral- to grid space with MIR bug.

        Args:
            ds_data: Dataset containing unfolded spectral coefficients.

        Returns:
            Dataset containing grid coefficients.
        """
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_mir(ds_data)

    def folded_spec_to_grid_grad_theta(
        self, ds_data: xr.Dataset, prefix: str = 'gt'
    ) -> xr.Dataset:
        """Applies the transformation from folded spectral- to grid space with gradient with respect to latitude.

        Args:
            ds_data: Dataset containing folded spectral coefficients.
            prefix: Prefix added in front of data variable names.

        Returns:
            Dataset containing grid coefficients.
        """
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid_grad_theta" transformation')
        ds_data = self.enforce_dtype(ds_data)
        new_names = (
            {var: f'{prefix}{var}' for var in ds_data} if prefix is not None else {}
        )
        ds_data = (
            xr.apply_ufunc(
                generic_folded_spec_to_grid_numpy,
                ds_data,
                self.grid.alm,
                kwargs=dict(
                    num_lon=self.num_lon,
                    grad_phi=False,
                    mir_bug=False,
                ),
                input_core_dims=[
                    ['c', 'l', 'm'],
                    ['latitude', 'l', 'm'],
                ],
                output_core_dims=[['latitude', 'longitude']],
                dask='parallelized',
                output_dtypes=[self.dtype],
                dask_gufunc_kwargs=dict(
                    output_sizes=dict(
                        longitude=self.num_lon,
                    )
                ),
            )
            .assign_coords(
                latitude=self.grid.lat,
                longitude=self.grid.lon,
            )
            .rename(**new_names)
        )
        return ds_data

    def unfolded_spec_to_grid_grad_theta(
        self, ds_data: xr.Dataset, prefix: str = 'gt'
    ) -> xr.Dataset:
        """Applies the transformation from unfolded spectral- to grid space with gradient with respect to latitude.

        Args:
            ds_data: Dataset containing unfolded spectral coefficients.
            prefix: Prefix added in front of data variable names.

        Returns:
            Dataset containing grid coefficients.
        """
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_grad_theta(ds_data, prefix)

    def folded_spec_to_grid_grad_phi(
        self, ds_data: xr.Dataset, prefix: str = 'gp'
    ) -> xr.Dataset:
        """Applies the transformation from folded spectral- to grid space with gradient with respect to longitude.

        Args:
            ds_data: Dataset containing folded spectral coefficients.
            prefix: Prefix added in front of data variable names.

        Returns:
            Dataset containing grid coefficients.
        """
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid_grad_phi" transformation')
        ds_data = self.enforce_dtype(ds_data)
        new_names = (
            {var: f'{prefix}{var}' for var in ds_data} if prefix is not None else {}
        )
        ds_data = (
            xr.apply_ufunc(
                generic_folded_spec_to_grid_numpy,
                ds_data,
                self.grid.plm,
                kwargs=dict(
                    num_lon=self.num_lon,
                    grad_phi=True,
                    mir_bug=False,
                ),
                input_core_dims=[
                    ['c', 'l', 'm'],
                    ['latitude', 'l', 'm'],
                ],
                output_core_dims=[['latitude', 'longitude']],
                dask='parallelized',
                output_dtypes=[self.dtype],
                dask_gufunc_kwargs=dict(
                    output_sizes=dict(
                        longitude=self.num_lon,
                    )
                ),
            )
            .assign_coords(
                latitude=self.grid.lat,
                longitude=self.grid.lon,
            )
            .rename(**new_names)
        )
        return ds_data

    def unfolded_spec_to_grid_grad_phi(
        self, ds_data: xr.Dataset, prefix: str = 'gp'
    ) -> xr.Dataset:
        """Applies the transformation from unfolded spectral- to grid space with gradient with respect to longitude.

        Args:
            ds_data: Dataset containing unfolded spectral coefficients.
            prefix: Prefix added in front of data variable names.

        Returns:
            Dataset containing grid coefficients.
        """
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_grad_phi(ds_data, prefix)

    def grid_to_folded_spec(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the transformation from grid- to folded spectral space.

        Args:
            ds_data: Dataset containing grid coefficients.

        Returns:
            Dataset containing folded spectral coefficients.
        """
        logger.info('applying "grid_to_folded_spec" transformation')
        ds_data = self.enforce_dtype(ds_data)
        ds_data = xr.apply_ufunc(
            grid_to_folded_spec_numpy,
            ds_data,
            self.grid.pw,
            input_core_dims=[
                ['latitude', 'longitude'],
                ['latitude', 'l', 'm'],
            ],
            output_core_dims=[['c', 'l', 'm']],
            dask='parallelized',
            output_dtypes=[self.dtype],
            dask_gufunc_kwargs=dict(
                output_sizes=dict(
                    c=2,
                )
            ),
        )
        return ds_data

    def grid_to_unfolded_spec(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the transformation from grid- to unfolded spectral space.

        Args:
            ds_data: Dataset containing grid coefficients.

        Returns:
            Dataset containing unfolded spectral coefficients.
        """
        ds_data = self.grid_to_folded_spec(ds_data)
        return self.unfold_clm(ds_data)

    def folded_spec_to_grid_full(
        self, ds_data: xr.Dataset, prefix_theta: str = 'gt', prefix_phi: str = 'gp'
    ) -> xr.Dataset:
        """Applies the transformation from folded spectral- to grid space with horizontal gradients.

        Args:
            ds_data: Dataset containing folded spectral coefficients.
            prefix_theta: Prefix added in front of data variable names for gradient with respect to latitude.
            prefix_phi: Prefix added in front of data variable names for gradient with respect to longitude.

        Returns:
            Dataset containing grid coefficients with horizontal gradients.
        """
        ds_grid_no_grad = self.folded_spec_to_grid(ds_data)
        ds_grid_grad_theta = self.folded_spec_to_grid_grad_theta(ds_data, prefix_theta)
        ds_grid_grad_phi = self.folded_spec_to_grid_grad_phi(ds_data, prefix_phi)
        return xr.merge((ds_grid_no_grad, ds_grid_grad_theta, ds_grid_grad_phi))

    def unfolded_spec_to_grid_full(
        self, ds_data: xr.Dataset, prefix_theta: str = 'gt', prefix_phi: str = 'gp'
    ) -> xr.Dataset:
        """Applies the transformation from unfolded spectral- to grid space with horizontal gradients.

        Args:
            ds_data: Dataset containing unfolded spectral coefficients.
            prefix_theta: Prefix added in front of data variable names for gradient with respect to latitude.
            prefix_phi: Prefix added in front of data variable names for gradient with respect to longitude.

        Returns:
            Dataset containing grid coefficients with horizontal gradients.
        """
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_full(ds_data, prefix_theta, prefix_phi)
