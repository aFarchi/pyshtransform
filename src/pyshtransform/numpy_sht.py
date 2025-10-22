#!/usr/bin/env python

import logging

import numpy as np
import xarray as xr

import pyshtransform.folding as ps_folding
import pyshtransform.misc as ps_misc
from pyshtransform.legendre import pre_glq, plmbar_d1

logger = logging.getLogger(__name__)


class NumpySphericalHarmonicsTransform(ps_folding.FoldingTransformation):

    def __init__(self, dtype, truncation, num_lat, num_lon, spline_order, num_splines, variant):
        super().__init__(dtype, truncation, factor=1)
        self.num_lat = num_lat
        self.num_lon = num_lon
        self.lat, self.lon = self.precompute_grid_nodes()
        self.plm, self.alm, self.pw = self.precompute_legendre_coefficients()
        self.wavelet_matrix = ps_misc.compute_wavelet_matrix(
            dtype=dtype,
            truncation=truncation,
            spline_order=spline_order,
            num_splines=num_splines,
        )
        self.variant = variant

    def apply(self, ds_data):
        return getattr(self, self.variant)(ds_data)

    def precompute_grid_nodes(self):
        cos_t, w = pre_glq(-1, 1, self.num_lat)
        lat = np.asin(cos_t) * 180 / np.pi
        lon = np.linspace(0, 360, self.num_lon, endpoint=False)
        return lat, lon

    def precompute_legendre_coefficients(self):
        cos_t, w = pre_glq(-1, 1, self.num_lat)
        cos_l = np.cos(self.lat * np.pi / 180)
        shape = (self.num_lat, self.truncation + 1, self.truncation + 1)
        plm = np.zeros(shape, dtype=self.dtype)
        alm = np.zeros(shape, dtype=self.dtype)
        pw = np.zeros(shape, dtype=self.dtype)
        indices_l, indices_m = np.tril_indices(self.truncation + 1)
        for i in range(self.num_lat):
            p, a = plmbar_d1(self.truncation, cos_t[i])
            plm[i, indices_m, indices_l] = p
            alm[i, indices_m, indices_l] = a * cos_l[i]
            pw[i, indices_m, indices_l] = 0.5 * w[i] * p
        return plm, alm, pw

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
            self.plm,
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
            dask_gufunc_kwargs=dict(output_sizes=dict(
                longitude=self.num_lon,
            )),
        ).assign_coords(
            latitude=self.lat,
            longitude=self.lon,
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
            self.plm,
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
            dask_gufunc_kwargs=dict(output_sizes=dict(
                longitude=self.num_lon,
            )),
        ).assign_coords(
            latitude=self.lat,
            longitude=self.lon,
        )

    def unfolded_spec_to_grid_mir(self, ds_data):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_mir(ds_data)

    def folded_spec_to_grid_grad_theta(self, ds_data, prefix='gt'):
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid_grad_theta" transformation')
        ds_data = self.enforce_dtype(ds_data)
        new_names = {
            var: f'{prefix}{var}'
            for var in ds_data
        } if prefix is not None else {}
        return xr.apply_ufunc(
            generic_folded_spec_to_grid_numpy,
            ds_data,
            self.alm,
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
            dask_gufunc_kwargs=dict(output_sizes=dict(
                longitude=self.num_lon,
            )),
        ).assign_coords(
            latitude=self.lat,
            longitude=self.lon,
        ).rename(**new_names)

    def unfolded_spec_to_grid_grad_theta(self, ds_data, prefix='gt'):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_grad_theta(ds_data, prefix)

    def folded_spec_to_grid_grad_phi(self, ds_data, prefix='gp'):
        ds_data = self.apply_wavelet_decomposition(ds_data)
        logger.info('applying "folded_spec_to_grid_grad_phi" transformation')
        ds_data = self.enforce_dtype(ds_data)
        new_names = {
            var: f'{prefix}{var}'
            for var in ds_data
        } if prefix is not None else {}
        return xr.apply_ufunc(
            generic_folded_spec_to_grid_numpy,
            ds_data,
            self.plm,
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
            dask_gufunc_kwargs=dict(output_sizes=dict(
                longitude=self.num_lon,
            )),
        ).assign_coords(
            latitude=self.lat,
            longitude=self.lon,
        ).rename(**new_names)

    def unfolded_spec_to_grid_grad_phi(self, ds_data, prefix='gp'):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_grad_phi(ds_data, prefix)

    def grid_to_folded_spec(self, ds_data):
        logger.info('applying "grid_to_folded_spec" transformation')
        ds_data = self.enforce_dtype(ds_data)
        return xr.apply_ufunc(
            grid_to_folded_spec_numpy,
            ds_data,
            self.pw,
            input_core_dims=[
                ['latitude', 'longitude'],
                ['latitude', 'l', 'm'],
            ],
            output_core_dims=[['c', 'l', 'm']],
            dask='parallelized',
            output_dtypes=[self.dtype],
            dask_gufunc_kwargs=dict(output_sizes=dict(
                c=2,
            )),
        )

    def grid_to_unfolded_spec(self, ds_data):
        ds_data = self.grid_to_folded_spec(ds_data)
        return self.unfold_clm(ds_data)

    def folded_spec_to_grid_full(self, ds_data, prefix_theta='gt', prefix_phi='gp'):
        ds_grid_no_grad = self.folded_spec_to_grid(ds_data)
        ds_grid_grad_theta = self.folded_spec_to_grid_grad_theta(ds_data, prefix_theta)
        ds_grid_grad_phi = self.folded_spec_to_grid_grad_phi(ds_data, prefix_phi)
        return xr.merge((
            ds_grid_no_grad,
            ds_grid_grad_theta,
            ds_grid_grad_phi
        ))

    def unfolded_spec_to_grid_full(self, ds_data, prefix_theta='gt', prefix_phi='gp'):
        ds_data = self.fold_clm(ds_data)
        return self.folded_spec_to_grid_full(ds_data, prefix_theta, prefix_phi)


def apply_wavelet_decomposition_numpy(f_clm, wavelet_matrix):
    return np.einsum(
        '...l,wl->...wl',
        f_clm,
        wavelet_matrix,
        casting='no',
    )


def apply_grad_phi_numpy(f_clm):
    shift = np.arange(f_clm.shape[-1])
    df_clm = np.zeros_like(f_clm)
    df_clm[..., 0, :, :] = shift * f_clm[..., 1, :, :]
    df_clm[..., 1, :, :] = - shift * f_clm[..., 0, :, :]
    return df_clm


def apply_mir_bug_numpy(f_clm):
    f_clm = f_clm.copy()
    f_clm[..., :, -1, -1] = 0
    return f_clm


def generic_folded_spec_to_grid_numpy(
        f_clm,
        plm,
        *,
        num_lon,
        grad_phi,
        mir_bug,
):
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


def grid_to_folded_spec_numpy(f, pw):
    # apply ihfft
    f_clm = np.fft.ihfft(f, axis=-1, norm='backward')
    # truncate the result
    f_clm = f_clm[..., :pw.shape[-1]]
    # move to real numbers
    f_clm = np.stack((f_clm.real, f_clm.imag), axis=-3)
    # apply inverse Legendre transformation
    return np.einsum('...im,iml->...lm', f_clm, pw, casting='no')
