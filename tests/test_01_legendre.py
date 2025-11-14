import common
import pytest
import xarray as xr

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform


@pytest.fixture(
    params=[
        {'truncation': 15, 'num_lat': 16, 'num_lon': 31},
        {'truncation': 31, 'num_lat': 32, 'num_lon': 64},
        {'truncation': 42, 'num_lat': 44, 'num_lon': 88},
        {'truncation': 63, 'num_lat': 128, 'num_lon': 256},
    ],
)
def config(request):
    return request.param


def test_legendre(config):
    ds_test = common.open_ds_01_legendre(**config)
    transformation = FullGridSphericalHarmonicsTransform(
        dtype='float64',
        unfolded_truncation=ds_test.truncation,
        folded_truncation=ds_test.truncation,
        num_lat=ds_test.num_lat,
        num_lon=ds_test.num_lon,
        spline_order=None,
        num_splines=None,
        variant=None,
    )
    ds_out = xr.Dataset(
        data_vars={
            'plm': (('latitude', 'm', 'l'), transformation.grid.plm),
            'alm': (('latitude', 'm', 'l'), transformation.grid.alm),
            'pw': (('latitude', 'm', 'l'), transformation.grid.pw),
        },
        coords={
            'latitude': ('latitude', transformation.grid.lat),
            'longitude': ('longitude', transformation.grid.lon),
        },
    )
    common.test_function(
        'legendre coefficients',
        ds_out,
        ds_test,
        rtol=1e-7,
        atol=0,
    )
