import common
import pytest

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform


@pytest.fixture(
    params=[
        dict(
            unfolded_truncation=1279,
            folded_truncation=7,
            num_lat=8,
            num_lon=16,
        ),
    ]
)
def config_success(request):
    return request.param


def construct_transformation(config, dtype):
    return FullGridSphericalHarmonicsTransform(
        **config,
        spline_order=None,
        num_splines=None,
        dtype=dtype,
        variant='',
    )


def test_precision_inverse_float64(ds_spec, config_success):
    transformation_float64 = construct_transformation(config_success, dtype='float64')
    ds_folded = transformation_float64.fold_clm(ds_spec)
    ds_grid_64 = transformation_float64.folded_spec_to_grid(ds_folded)
    ds_folded_64 = transformation_float64.grid_to_folded_spec(ds_grid_64)
    common.test_function(
        'float64, grid_to_spec o spec_to_grid = Id',
        ds_folded_64,
        ds_folded,
        rtol=1e-6,
        atol=0,
    )


def test_precision_forward_float32(ds_spec, config_success):
    transformation_float64 = construct_transformation(config_success, dtype='float64')
    transformation_float32 = construct_transformation(config_success, dtype='float32')
    ds_folded = transformation_float64.fold_clm(ds_spec)
    ds_grid_64 = transformation_float64.folded_spec_to_grid(ds_folded)
    ds_grid_32 = transformation_float32.folded_spec_to_grid(ds_folded)
    common.test_function(
        'float32 vs float64, spec_to_grid',
        ds_grid_32,
        ds_grid_64,
        rtol=1e-6,
        atol=0,
    )


def test_precision_inverse_float32(ds_spec, config_success):
    transformation_float64 = construct_transformation(config_success, dtype='float64')
    transformation_float32 = construct_transformation(config_success, dtype='float32')
    ds_folded = transformation_float64.fold_clm(ds_spec)
    ds_grid_64 = transformation_float64.folded_spec_to_grid(ds_folded)
    ds_folded_64 = transformation_float64.grid_to_folded_spec(ds_grid_64)
    ds_folded_32 = transformation_float32.grid_to_folded_spec(ds_grid_64)
    common.test_function(
        'float32 vs float64, grid_to_spec',
        ds_folded_32,
        ds_folded_64,
        rtol=1e-2,
        atol=0,
    )
