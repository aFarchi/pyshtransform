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
    def __init__(
        self, dtype, truncation, num_lat, num_lon, spline_order, num_splines, variant
    ):
        super().__init__(dtype=dtype, truncation=truncation, factor=1)
        self.num_lat = num_lat
        self.num_lon = num_lon
        self.grid = FullGrid(
            dtype=dtype,
            truncation=truncation,
            num_lat=num_lat,
            num_lon=num_lon,
        )
        self.wavelet_matrix = compute_wavelet_matrix(
            dtype=dtype,
            truncation=truncation,
            spline_order=spline_order,
            num_splines=num_splines,
        )
        self.variant = variant

    def apply(self, ds_data):
        return getattr(self, self.variant)(ds_data)

    def apply_wavelet_decomposition(self, ds_data):
        if self.wavelet_matrix is None:
            return ds_data
        logger.info('applying "wavelet_decomposition" transformation')
        ds_data = self.enforce_dtype(ds_data)
        return xr.apply_ufunc(
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

    def folded_spec_to_grid(self, ds_data):
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid" transformation')
        ds_data = self.enforce_dtype(ds_data)
        return xr.apply_ufunc(
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

    def unfolded_spec_to_grid(self, ds_data):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid(ds_data)

    def folded_spec_to_grid_mir(self, ds_data):
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid_mir" transformation')
        ds_data = self.enforce_dtype(ds_data)
        return xr.apply_ufunc(
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

    def unfolded_spec_to_grid_mir(self, ds_data):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_mir(ds_data)

    def folded_spec_to_grid_grad_theta(self, ds_data, prefix='gt'):
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid_grad_theta" transformation')
        ds_data = self.enforce_dtype(ds_data)
        new_names = (
            {var: f'{prefix}{var}' for var in ds_data} if prefix is not None else {}
        )
        return (
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

    def unfolded_spec_to_grid_grad_theta(self, ds_data, prefix='gt'):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_grad_theta(ds_data, prefix)

    def folded_spec_to_grid_grad_phi(self, ds_data, prefix='gp'):
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid_grad_phi" transformation')
        ds_data = self.enforce_dtype(ds_data)
        new_names = (
            {var: f'{prefix}{var}' for var in ds_data} if prefix is not None else {}
        )
        return (
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

    def unfolded_spec_to_grid_grad_phi(self, ds_data, prefix='gp'):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_grad_phi(ds_data, prefix)

    def grid_to_folded_spec(self, ds_data):
        logger.info('applying "grid_to_folded_spec" transformation')
        ds_data = self.enforce_dtype(ds_data)
        return xr.apply_ufunc(
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

    def grid_to_unfolded_spec(self, ds_data):
        ds_data = self.grid_to_folded_spec(ds_data)
        return self.unfold_clm(ds_data)

    def folded_spec_to_grid_full(self, ds_data, prefix_theta='gt', prefix_phi='gp'):
        ds_grid_no_grad = self.folded_spec_to_grid(ds_data)
        ds_grid_grad_theta = self.folded_spec_to_grid_grad_theta(ds_data, prefix_theta)
        ds_grid_grad_phi = self.folded_spec_to_grid_grad_phi(ds_data, prefix_phi)
        return xr.merge((ds_grid_no_grad, ds_grid_grad_theta, ds_grid_grad_phi))

    def unfolded_spec_to_grid_full(self, ds_data, prefix_theta='gt', prefix_phi='gp'):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_full(ds_data, prefix_theta, prefix_phi)
