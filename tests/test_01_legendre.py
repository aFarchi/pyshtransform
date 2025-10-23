#!/usr/bin/env python

import common
import pytest
import xarray as xr

import pyshtransform.numpy_sht as np_sht


@pytest.fixture(
    params=[
        dict(
            truncation=15,
            num_lat=16,
            num_lon=31,
        ),
        dict(
            truncation=31,
            num_lat=32,
            num_lon=64,
        ),
        dict(
            truncation=42,
            num_lat=44,
            num_lon=88,
        ),
        dict(
            truncation=63,
            num_lat=128,
            num_lon=256,
        ),
    ]
)
def config(request):
    return request.param


def test_legendre(config):
    ds_test = common.open_ds_01_legendre(**config)
    transformation = np_sht.NumpySphericalHarmonicsTransform(
        truncation=ds_test.truncation,
        num_lat=ds_test.num_lat,
        num_lon=ds_test.num_lon,
        spline_order=None,
        num_splines=None,
        dtype='float64',
        variant=None,
    )
    transformation.precompute_folding_coefficients(ds_test.truncation)
    ds_out = xr.Dataset(
        data_vars=dict(
            plm=(('latitude', 'm', 'l'), transformation.plm),
            alm=(('latitude', 'm', 'l'), transformation.alm),
            pw=(('latitude', 'm', 'l'), transformation.pw),
        ),
        coords=dict(
            latitude=('latitude', transformation.lat),
            longitude=('longitude', transformation.lon),
        ),
    )
    common.test_function(
        'legendre coefficients',
        ds_out,
        ds_test,
        rtol=1e-7,
        atol=0,
    )
