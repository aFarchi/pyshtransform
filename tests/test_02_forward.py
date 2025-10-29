import common
import pytest

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform


def construct_transformation(ds_grid_ref):
    return FullGridSphericalHarmonicsTransform(
        dtype='float64',
        unfolded_truncation=1279,
        folded_truncation=ds_grid_ref.truncation,
        num_lat=len(ds_grid_ref.latitude),
        num_lon=len(ds_grid_ref.longitude),
        spline_order=None,
        num_splines=None,
        variant='',
    )


def test_forward_mir(ds_spec, ds_grid_ref):
    transformation = construct_transformation(ds_grid_ref)
    ds_grid = transformation.unfolded_spec_to_grid_mir(ds_spec)
    common.test_function(
        'unfolded_spec_to_grid_mir',
        ds_grid,
        ds_grid_ref,
        rtol=1e-5,
        atol=0,
    )


def test_fail_forward(ds_spec, ds_grid_ref):
    transformation = construct_transformation(ds_grid_ref)
    ds_grid = transformation.unfolded_spec_to_grid(ds_spec)
    with pytest.raises(AssertionError) as _e_info:
        common.test_function(
            '*fail* unfolded_spec_to_grid',
            ds_grid,
            ds_grid_ref,
            rtol=1e-5,
            atol=0,
        )
