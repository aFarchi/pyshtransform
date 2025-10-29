import common
import pytest

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform


@pytest.fixture(
    params=[
        dict(
            unfolded_truncation=1279,
            folded_truncation=63,
            num_lat=64,
            num_lon=128,
        ),
        dict(
            unfolded_truncation=1279,
            folded_truncation=191,
            num_lat=192,
            num_lon=384,
        ),
        dict(
            unfolded_truncation=1279,
            folded_truncation=15,
            num_lat=16,
            num_lon=31,
        ),
        dict(
            unfolded_truncation=1279,
            folded_truncation=15,
            num_lat=16,
            num_lon=63,
        ),
        dict(
            unfolded_truncation=1279,
            folded_truncation=15,
            num_lat=32,
            num_lon=31,
        ),
    ]
)
def config_success(request):
    return request.param


@pytest.fixture(
    params=[
        dict(
            unfolded_truncation=1279,
            folded_truncation=15,
            num_lat=14,
            num_lon=31,
        ),
    ]
)
def config_fail(request):
    return request.param


def construct_transformation(config):
    return FullGridSphericalHarmonicsTransform(
        **config,
        spline_order=None,
        num_splines=None,
        dtype='float64',
        variant='',
    )


def test_inverse_left(ds_spec, config_success):
    transformation = construct_transformation(config_success)
    ds_folded = transformation.fold_clm(ds_spec)
    ds_grid = transformation.unfolded_spec_to_grid(ds_spec)
    ds_folded_inverse = transformation.grid_to_folded_spec(ds_grid)
    common.test_function(
        'grid_to_spec o spec_to_grid = Id',
        ds_folded_inverse,
        ds_folded,
        rtol=1e-6,
        atol=0,
    )


def test_fail_inverse_left(ds_spec, config_fail):
    transformation = construct_transformation(config_fail)
    ds_folded = transformation.fold_clm(ds_spec)
    ds_grid = transformation.unfolded_spec_to_grid(ds_spec)
    ds_folded_inverse = transformation.grid_to_folded_spec(ds_grid)
    with pytest.raises(AssertionError) as _e_info:
        common.test_function(
            '*fail* grid_to_spec o spec_to_grid = Id',
            ds_folded_inverse,
            ds_folded,
            rtol=1e-6,
            atol=0,
        )


def test_inverse_right(ds_spec, config_success):
    transformation = construct_transformation(config_success)
    ds_grid = transformation.unfolded_spec_to_grid(ds_spec)
    ds_folded_inverse = transformation.grid_to_folded_spec(ds_grid)
    ds_grid_inverse = transformation.folded_spec_to_grid(ds_folded_inverse)
    common.test_function(
        'spec_to_grid o grid_to_spec o spec_to_grid = spec_to_grid',
        ds_grid_inverse,
        ds_grid,
        rtol=1e-6,
        atol=0,
    )


def test_fail_inverse_right(ds_spec, config_fail):
    transformation = construct_transformation(config_fail)
    ds_grid = transformation.unfolded_spec_to_grid(ds_spec)
    ds_folded_inverse = transformation.grid_to_folded_spec(ds_grid)
    ds_grid_inverse = transformation.folded_spec_to_grid(ds_folded_inverse)
    with pytest.raises(AssertionError) as _e_info:
        common.test_function(
            '*fail* spec_to_grid o grid_to_spec o spec_to_grid = spec_to_grid',
            ds_grid_inverse,
            ds_grid,
            rtol=1e-6,
            atol=0,
        )
