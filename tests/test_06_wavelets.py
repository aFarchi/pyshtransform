import common
import pytest

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform


@pytest.fixture(
    params=[
        dict(
            truncation=15,
            spline_order=1,
            num_splines=3,
        ),
    ]
)
def config(request):
    return request.param


@pytest.fixture
def ds_06_wavelets(config):
    return common.open_ds_06_wavelets(**config)


def construct_transformation(ds_06_wavelets):
    return FullGridSphericalHarmonicsTransform(
        dtype='float64',
        unfolded_truncation=1279,
        folded_truncation=ds_06_wavelets.truncation,
        num_lat=len(ds_06_wavelets.latitude),
        num_lon=len(ds_06_wavelets.longitude),
        spline_order=ds_06_wavelets.spline_order,
        num_splines=ds_06_wavelets.num_splines,
        variant='',
    )


def test_forward(ds_spec, ds_06_wavelets):
    transformation = construct_transformation(ds_06_wavelets)
    ds_spec = ds_spec.load()
    ds_grid = transformation.unfolded_spec_to_grid(ds_spec)
    common.test_function(
        'spec_to_grid',
        ds_grid,
        ds_06_wavelets,
        rtol=1e-7,
        atol=0,
    )
