import common  # ruff: ignore[implicit-namespace-package]
import pytest

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform


@pytest.fixture(
    params=[
        {
            'folded_truncation': 7,
            'num_lat': 8,
            'num_lon': 16,
        },
    ],
)
def config_success(request):
    return request.param


def construct_transformation(config, dtype, ds_spec):
    return FullGridSphericalHarmonicsTransform(
        **(config | {'unfolded_truncation': ds_spec.truncation}),
        spline_order=None,
        num_splines=None,
        dtype=dtype,
        variant='',
    )


def test_precision_inverse_float64(ds_spec, config_success):
    transformation_float64 = construct_transformation(
        config_success,
        dtype='float64',
        ds_spec=ds_spec,
    )
    ds_folded = transformation_float64.fold_clm(ds_spec)
    ds_grid_64 = transformation_float64.folded_spec_to_grid(ds_folded)
    ds_folded_64 = transformation_float64.grid_to_folded_spec(ds_grid_64)
    for name, atol in {
        't': 2e-11,
        'vo': 1e-18,
        'd': 1e-18,
        'u': 1e-12,
        'v': 1e-12,
    }.items():
        common.assert_spectral_error(
            ds_folded_64[[name]],
            ds_folded[[name]],
            atol=atol,
            bulk_rtol=1e-10,
        )


def test_precision_forward_float32(ds_spec, config_success):
    transformation_float64 = construct_transformation(
        config_success,
        dtype='float64',
        ds_spec=ds_spec,
    )
    transformation_float32 = construct_transformation(
        config_success,
        dtype='float32',
        ds_spec=ds_spec,
    )
    ds_folded = transformation_float64.fold_clm(ds_spec)
    ds_grid_64 = transformation_float64.folded_spec_to_grid(ds_folded)
    ds_grid_32 = transformation_float32.folded_spec_to_grid(ds_folded)
    for name, atol in {
        't': 1e-4,
        'vo': 1e-12,
        'd': 1e-12,
        'u': 2e-6,
        'v': 2e-6,
    }.items():
        common.assert_grid_error(
            ds_grid_32[[name]],
            ds_grid_64[[name]],
            transformation_float64.grid.gauss_weights,
            atol=atol,
            rms_rtol=1e-6,
            max_rtol=1e-6,
            polar_rtol=1e-6,
        )


def test_precision_inverse_float32(ds_spec, config_success):
    transformation_float64 = construct_transformation(
        config_success,
        dtype='float64',
        ds_spec=ds_spec,
    )
    transformation_float32 = construct_transformation(
        config_success,
        dtype='float32',
        ds_spec=ds_spec,
    )
    ds_folded = transformation_float64.fold_clm(ds_spec)
    ds_grid_64 = transformation_float64.folded_spec_to_grid(ds_folded)
    ds_folded_64 = transformation_float64.grid_to_folded_spec(ds_grid_64)
    ds_folded_32 = transformation_float32.grid_to_folded_spec(ds_grid_64)
    for name, atol in {
        't': 1e-4,
        'vo': 1e-12,
        'd': 1e-12,
        'u': 1e-6,
        'v': 1e-6,
    }.items():
        common.assert_spectral_error(
            ds_folded_32[[name]],
            ds_folded_64[[name]],
            atol=atol,
            bulk_rtol=1e-4,
        )
