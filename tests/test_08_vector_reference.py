# ruff: file-ignore[assert, implicit-namespace-package]
import common
import numpy as np
import pytest
import xarray as xr

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform

SYNTHESIS_BUDGETS = {
    63: {'atol': 1e-6, 'bulk_rtol': 5e-4, 'tail_rtol': 6e-4, 'total_rtol': 1e-4},
    191: {'atol': 1e-6, 'bulk_rtol': 9e-4, 'tail_rtol': 1.2e-3, 'total_rtol': 1e-4},
}
ANALYSIS_BUDGETS = {
    63: {
        'vo': {'atol': 1e-18, 'bulk_rtol': 3e-4, 'tail_rtol': 4e-4, 'total_rtol': 2e-4},
        'd': {
            'atol': 1e-18,
            'bulk_rtol': 8e-4,
            'tail_rtol': 1.1e-3,
            'total_rtol': 6e-4,
        },
    },
    191: {
        'vo': {'atol': 1e-18, 'bulk_rtol': 8e-4, 'tail_rtol': 9e-4, 'total_rtol': 3e-4},
        'd': {
            'atol': 1e-18,
            'bulk_rtol': 1.2e-3,
            'tail_rtol': 1.4e-3,
            'total_rtol': 8e-4,
        },
    },
}


@pytest.fixture
def reference_case(ds_spec, ds_grid_ref):
    transform = FullGridSphericalHarmonicsTransform(
        dtype='float64',
        unfolded_truncation=int(ds_spec.truncation),
        folded_truncation=int(ds_grid_ref.truncation),
        num_lat=ds_grid_ref.sizes['latitude'],
        num_lon=ds_grid_ref.sizes['longitude'],
        spline_order=None,
        num_splines=None,
        variant=None,
    )
    np.testing.assert_allclose(
        ds_grid_ref.latitude,
        transform.grid.lat,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ds_grid_ref.longitude,
        transform.grid.lon,
        rtol=0,
        atol=1e-12,
    )
    reference = ds_grid_ref.assign_coords(
        latitude=transform.grid.lat,
        longitude=transform.grid.lon,
    )
    return transform, transform.fold_clm(ds_spec), reference


def test_fixture_contract(ds_spec, ds_grid_ref):
    assert set(ds_spec) == set(ds_grid_ref) == {'t', 'vo', 'd', 'u', 'v'}
    np.testing.assert_array_equal(ds_spec.level, [96, 137])
    np.testing.assert_array_equal(ds_grid_ref.level, ds_spec.level)
    truncation = int(ds_spec.truncation)
    assert ds_spec.sizes['clm'] == (truncation + 1) * (truncation + 2)
    assert ds_grid_ref.sizes['latitude'] == int(ds_grid_ref.truncation) + 1
    assert ds_grid_ref.sizes['longitude'] == 2 * ds_grid_ref.sizes['latitude']
    assert np.all(np.diff(ds_grid_ref.latitude) < 0)
    for name in ds_spec:
        assert ds_spec[name].dims == ('level', 'clm')
        assert ds_grid_ref[name].dims == ('level', 'latitude', 'longitude')
        assert bool(np.isfinite(ds_spec[name]).all().compute())
        assert bool(np.isfinite(ds_grid_ref[name]).all().compute())


@pytest.mark.parametrize('level', [96, 137])
def test_vector_synthesis_reference(reference_case, level):
    transform, folded, reference = reference_case
    expected = reference[['u', 'v']].sel(level=level)
    actual = transform.folded_spec_vod_to_grid_uv(folded[['vo', 'd']].sel(level=level))
    common.assert_grid_error(
        actual,
        expected,
        transform.grid.gauss_weights,
        atol=1e-6,
        rms_rtol=1e-4,
        max_rtol=3e-4,
        polar_rtol=3e-4,
    )
    common.assert_grid_spectrum(
        transform,
        actual,
        expected,
        **SYNTHESIS_BUDGETS[transform.folded_truncation],
    )


@pytest.mark.parametrize('level', [96, 137])
def test_vector_analysis_reference(reference_case, level):
    transform, folded, reference = reference_case
    actual = transform.grid_uv_to_folded_spec_vod(
        reference[['u', 'v']].sel(level=level),
    )
    expected = folded[['vo', 'd']].sel(level=level)
    for name, budget in ANALYSIS_BUDGETS[transform.folded_truncation].items():
        common.assert_spectral_error(actual[[name]], expected[[name]], **budget)


@pytest.mark.parametrize('level', [96, 137])
@pytest.mark.parametrize('name', ['t', 'vo', 'd', 'u', 'v'])
def test_scalar_reference_spectra(reference_case, level, name):
    transform, folded, reference = reference_case
    expected = reference[[name]].sel(level=level)
    if name in {'u', 'v'}:
        expected *= xr.DataArray(transform.grid.cos_lat, dims='latitude')
    actual = transform.folded_spec_to_grid_mir(folded[[name]].sel(level=level))
    if name == 't':
        spectrum = {
            'atol': 1e-5,
            'bulk_rtol': 1e-3,
            'tail_rtol': 1.2e-3,
            'total_rtol': 3e-6,
        }
        spatial = {'atol': 1e-5, 'rms_rtol': 3e-6, 'max_rtol': 8e-6, 'polar_rtol': 8e-6}
    elif name in {'vo', 'd'}:
        spectrum = {
            'atol': 1e-10,
            'bulk_rtol': 3e-4,
            'tail_rtol': 4e-4,
            'total_rtol': 2e-4,
        }
        spatial = {
            'atol': 1e-10,
            'rms_rtol': 2.5e-4,
            'max_rtol': 7e-4,
            'polar_rtol': 7e-4,
        }
    else:
        coarse = transform.folded_truncation == 63
        spectrum = {
            'atol': 1e-6,
            'bulk_rtol': 1.2e-3,
            'tail_rtol': 1.2,
            'total_rtol': 0.03 if coarse else 0.01,
            'tail_start': transform.folded_truncation,
        }
        spatial = {
            'atol': 1e-6,
            'rms_rtol': 0.04 if coarse else 0.012,
            'max_rtol': 0.16 if coarse else 0.07,
            'polar_rtol': 0.16 if coarse else 0.07,
        }
    common.assert_grid_spectrum(transform, actual, expected, **spectrum)
    common.assert_grid_error(actual, expected, transform.grid.gauss_weights, **spatial)


def test_mir_highest_mode_omission(reference_case):
    transform, folded, _reference = reference_case
    ordinary = transform.folded_spec_to_grid(folded[['t', 'vo', 'd']])
    mir = transform.folded_spec_to_grid_mir(folded[['t', 'vo', 'd']])
    isolated = folded[['t', 'vo', 'd']].compute().copy(deep=True)
    for name in isolated:
        highest = isolated[name].values[..., :, -1, -1].copy()
        isolated[name].values[:] = 0
        isolated[name].values[..., :, -1, -1] = highest
    contribution = transform.folded_spec_to_grid(isolated)
    xr.testing.assert_allclose(ordinary - mir, contribution, rtol=1e-8, atol=1e-12)
    assert float(abs(contribution.t).max().compute()) > 0


@pytest.mark.parametrize('level', [96, 137])
def test_reference_wind_projection(reference_case, level):
    transform, _folded, reference = reference_case
    expected = reference[['u', 'v']].sel(level=level)
    projected = transform.grid_uv_to_folded_spec_vod(expected)
    reconstructed = transform.folded_spec_vod_to_grid_uv(projected)
    recovered = transform.grid_uv_to_folded_spec_vod(reconstructed)
    common.assert_spectral_error(recovered, projected, atol=1e-18, bulk_rtol=1e-10)
    common.assert_grid_error(
        reconstructed,
        expected,
        transform.grid.gauss_weights,
        atol=1e-6,
        rms_rtol=1e-3,
        max_rtol=3e-3,
        polar_rtol=3e-3,
    )
