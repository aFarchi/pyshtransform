import common  # ruff: ignore[implicit-namespace-package]

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform


def construct_transformation(ds_grid_ref, ds_spec):
    return FullGridSphericalHarmonicsTransform(
        dtype='float64',
        unfolded_truncation=ds_spec.truncation,
        folded_truncation=ds_grid_ref.truncation,
        num_lat=len(ds_grid_ref.latitude),
        num_lon=len(ds_grid_ref.longitude),
        spline_order=None,
        num_splines=None,
        variant='',
    )


def test_forward_mir(ds_spec, ds_grid_ref):
    transformation = construct_transformation(ds_grid_ref, ds_spec)
    ds_grid = transformation.unfolded_spec_to_grid_mir(ds_spec[['t']])
    common.test_function(
        'unfolded_spec_to_grid_mir',
        ds_grid,
        ds_grid_ref[['t']],
        rtol=1e-5,
        atol=0,
    )
