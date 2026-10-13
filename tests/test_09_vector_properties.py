# ruff: file-ignore[assert, implicit-namespace-package]
import common
import numpy as np
import pytest
import xarray as xr

RADIUS = 6_371_229.0


@pytest.mark.parametrize('branch', ['vo', 'd'])
def test_analytic_zonal_flow(branch):
    transform = common.construct_transform(7)
    latitude = np.deg2rad(transform.grid.lat)
    amplitude = 20.0
    wind = np.broadcast_to(
        amplitude * np.cos(latitude)[:, None],
        (transform.num_lat, transform.num_lon),
    ).copy()
    grid = xr.Dataset(
        {
            'u': (
                ('latitude', 'longitude'),
                wind if branch == 'vo' else np.zeros_like(wind),
            ),
            'v': (
                ('latitude', 'longitude'),
                wind if branch == 'd' else np.zeros_like(wind),
            ),
        },
        coords={'latitude': transform.grid.lat, 'longitude': transform.grid.lon},
    )
    expected = common.zero_vod(7)
    expected[branch].values[0, 1, 0] = (
        (1 if branch == 'vo' else -1) * 2 * amplitude / (RADIUS * np.sqrt(3))
    )
    analyzed = transform.grid_uv_to_folded_spec_vod(grid)
    common.assert_spectral_error(analyzed, expected, atol=1e-19, bulk_rtol=1e-12)
    synthesized = transform.folded_spec_vod_to_grid_uv(expected)
    common.assert_grid_error(
        synthesized,
        grid,
        transform.grid.gauss_weights,
        atol=1e-12,
        rms_rtol=1e-12,
        max_rtol=1e-12,
        polar_rtol=1e-12,
    )


def test_degree_power_normalization_and_mask():
    coefficients = common.zero_vod(3)
    coefficients.vo.values[0, 0, 0] = 3
    coefficients.vo.values[:, 2, 1] = [3, 4]
    coefficients.vo.values[0, 0, 3] = 100
    np.testing.assert_array_equal(common.degree_power(coefficients).vo, [9, 0, 50, 0])


def test_spectral_assertion_detects_tail_and_zero_reference():
    reference = common.zero_vod(7).expand_dims(level=[96, 137]).copy(deep=True)
    actual = reference.copy(deep=True)
    actual.vo.values[1, 0, 7, 7] = 1e-5
    with pytest.raises(AssertionError, match=r'vo.*level.*137.*l.*7'):
        common.assert_spectral_error(actual, reference, atol=1e-12, bulk_rtol=1e-6)
    common.assert_spectral_error(reference, reference, atol=0, bulk_rtol=0)


def test_spectral_assertion_rejects_coordinate_mismatch():
    reference = common.zero_vod(3).expand_dims(level=[96, 137])
    actual = reference.assign_coords(level=[95, 137])
    with pytest.raises(AssertionError):
        common.assert_spectral_error(actual, reference, atol=0, bulk_rtol=0)


@pytest.fixture(params=[(8, 15), (8, 16), (16, 32)])
def transform(request):
    num_lat, num_lon = request.param
    return common.construct_transform(7, num_lat=num_lat, num_lon=num_lon)


@pytest.fixture
def mixed_vod():
    coefficients = common.zero_vod(7).expand_dims(level=[96, 137]).copy(deep=True)
    generator = np.random.default_rng(20261013)
    valid = np.tri(8, dtype=bool)
    for name in coefficients:
        coefficients[name].values[:] = generator.normal(size=(2, 2, 8, 8)) * 1e-6
        coefficients[name].values[..., ~valid] = 0
        coefficients[name].values[:, 1, :, 0] = 0
        coefficients[name].values[:, :, 0, 0] = 0
    return coefficients


@pytest.mark.parametrize('branch', ['vo', 'd'])
@pytest.mark.parametrize(
    ('degree', 'order', 'component'),
    [
        (1, 0, 0),
        (3, 1, 0),
        (3, 1, 1),
        (4, 4, 0),
        (4, 4, 1),
        (7, 0, 0),
        (7, 3, 0),
        (7, 3, 1),
        (7, 7, 0),
        (7, 7, 1),
    ],
)
def test_isolated_vector_mode(transform, branch, degree, order, component):
    coefficients = common.zero_vod(7)
    coefficients[branch].values[component, degree, order] = 1e-6
    winds = transform.folded_spec_vod_to_grid_uv(coefficients)
    recovered = transform.grid_uv_to_folded_spec_vod(winds)
    common.assert_spectral_error(recovered, coefficients, atol=1e-19, bulk_rtol=1e-12)


def test_vector_round_trips(transform, mixed_vod):
    winds = transform.folded_spec_vod_to_grid_uv(mixed_vod)
    coefficients = transform.grid_uv_to_folded_spec_vod(winds)
    common.assert_spectral_error(coefficients, mixed_vod, atol=1e-19, bulk_rtol=1e-12)
    reconstructed = transform.folded_spec_vod_to_grid_uv(coefficients)
    common.assert_grid_error(
        reconstructed,
        winds,
        transform.grid.gauss_weights,
        atol=1e-12,
        rms_rtol=1e-12,
        max_rtol=1e-11,
        polar_rtol=1e-11,
    )


def test_zero_and_degree_zero_modes(transform):
    zero = common.zero_vod(7)
    winds = transform.folded_spec_vod_to_grid_uv(zero)
    for field in winds.values():
        np.testing.assert_array_equal(field, 0)
    xr.testing.assert_equal(transform.grid_uv_to_folded_spec_vod(winds), zero)
    null = zero.copy(deep=True)
    null.vo.values[:, 0, 0] = [1, 2]
    null.d.values[:, 0, 0] = [3, 4]
    xr.testing.assert_equal(transform.folded_spec_vod_to_grid_uv(null), winds)


def test_invalid_triangle_is_ignored(transform, mixed_vod):
    dirty = mixed_vod.copy(deep=True)
    invalid = ~np.tri(8, dtype=bool)
    for name in dirty:
        dirty[name].values[..., invalid] = np.nan
    xr.testing.assert_equal(
        transform.folded_spec_vod_to_grid_uv(dirty),
        transform.folded_spec_vod_to_grid_uv(mixed_vod),
    )


def test_analysis_storage_and_global_means(transform):
    generator = np.random.default_rng(17)
    winds = xr.Dataset({
        name: (
            ('latitude', 'longitude'),
            generator.normal(size=(transform.num_lat, transform.num_lon)),
        )
        for name in ('u', 'v')
    })
    coefficients = transform.grid_uv_to_folded_spec_vod(winds)
    for field in coefficients.values():
        np.testing.assert_array_equal(field.values[..., ~np.tri(8, dtype=bool)], 0)
        np.testing.assert_array_equal(field.isel(c=1, m=0), 0)
        np.testing.assert_array_equal(field.isel(l=0, m=0), 0)


@pytest.mark.parametrize('direction', ['synthesis', 'analysis'])
def test_linearity_and_no_mutation(transform, mixed_vod, direction):
    operation = (
        transform.folded_spec_vod_to_grid_uv
        if direction == 'synthesis'
        else transform.grid_uv_to_folded_spec_vod
    )
    first = (
        mixed_vod
        if direction == 'synthesis'
        else transform.folded_spec_vod_to_grid_uv(mixed_vod)
    )
    second = first.isel(level=[1, 0]).assign_coords(level=first.level)
    original = first.copy(deep=True)
    combined = operation(2 * first - 3 * second)
    expected = 2 * operation(first) - 3 * operation(second)
    xr.testing.assert_allclose(
        combined,
        expected,
        rtol=1e-12,
        atol=1e-12 if direction == 'synthesis' else 1e-19,
    )
    xr.testing.assert_identical(first, original)


@pytest.mark.parametrize('direction', ['synthesis', 'analysis'])
def test_batching_dimension_order_and_dask(transform, mixed_vod, direction):
    operation = (
        transform.folded_spec_vod_to_grid_uv
        if direction == 'synthesis'
        else transform.grid_uv_to_folded_spec_vod
    )
    source = (
        mixed_vod
        if direction == 'synthesis'
        else transform.folded_spec_vod_to_grid_uv(mixed_vod)
    )
    source = source.expand_dims(member=[0, 1])
    expected = operation(source)
    reordered = operation(source.transpose(*reversed(tuple(source.dims))))
    xr.testing.assert_allclose(reordered.transpose(*expected.dims), expected)
    lazy = operation(source.chunk(level=1, member=1))
    assert all(hasattr(field.data, 'dask') for field in lazy.values())
    xr.testing.assert_allclose(lazy.compute(), expected)
    for level in source.level.values:
        xr.testing.assert_allclose(
            operation(source.sel(level=level)),
            expected.sel(level=level),
        )
    assert set(expected) == ({'u', 'v'} if direction == 'synthesis' else {'vo', 'd'})
    xr.testing.assert_equal(expected.level, source.level)
    xr.testing.assert_equal(expected.member, source.member)
    if direction == 'synthesis':
        np.testing.assert_array_equal(expected.latitude, transform.grid.lat)
        np.testing.assert_array_equal(expected.longitude, transform.grid.lon)


@pytest.mark.parametrize('direction', ['synthesis', 'analysis'])
def test_vector_float32_accuracy(mixed_vod, direction):
    precise = common.construct_transform(7)
    reduced = common.construct_transform(7, dtype='float32')
    source = (
        mixed_vod
        if direction == 'synthesis'
        else precise.folded_spec_vod_to_grid_uv(mixed_vod)
    )
    method = (
        'folded_spec_vod_to_grid_uv'
        if direction == 'synthesis'
        else 'grid_uv_to_folded_spec_vod'
    )
    expected = getattr(precise, method)(source)
    actual = getattr(reduced, method)(source)
    if direction == 'analysis':
        common.assert_spectral_error(actual, expected, atol=1e-12, bulk_rtol=2e-6)
    else:
        common.assert_grid_error(
            actual,
            expected,
            precise.grid.gauss_weights,
            atol=1e-5,
            rms_rtol=2e-6,
            max_rtol=2e-5,
            polar_rtol=2e-5,
        )


def test_fold_unfold_vector_conventions(mixed_vod):
    transform = common.construct_transform(7)
    unfolded = transform.unfold_clm(mixed_vod)
    xr.testing.assert_allclose(transform.fold_clm(unfolded), mixed_vod, rtol=0, atol=0)
    isolated = common.zero_vod(7)
    isolated.vo.values[1, 3, 1] = 2
    flat = transform.unfold_clm(isolated).vo.values
    offset = 2 * (8 + 3 - 1)
    assert flat[offset + 1] == -2
    assert np.count_nonzero(flat) == 1


@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('direction', ['synthesis', 'analysis'])
@pytest.mark.parametrize(
    'dtype',
    [
        'float64',
        pytest.param(
            'float32',
            marks=pytest.mark.xfail(
                strict=True,
                reason='Vector kernels promote float32 inputs to float64 output',
            ),
        ),
    ],
)
def test_vector_output_dtype(mixed_vod, dtype, direction, lazy):
    transform = common.construct_transform(7, dtype=dtype)
    source = (
        mixed_vod
        if direction == 'synthesis'
        else transform.folded_spec_vod_to_grid_uv(mixed_vod)
    )
    if lazy:
        source = source.chunk(level=1)
    method = (
        'folded_spec_vod_to_grid_uv'
        if direction == 'synthesis'
        else 'grid_uv_to_folded_spec_vod'
    )
    actual = getattr(transform, method)(source).compute()
    for field in actual.values():
        assert field.dtype == np.dtype(dtype)


def test_spectral_assertion_detects_equal_power_phase_error():
    reference = common.zero_vod(7)
    reference.vo.values[0, 3, 2] = 1e-6
    actual = -reference
    xr.testing.assert_equal(common.degree_power(actual), common.degree_power(reference))
    with pytest.raises(AssertionError, match='vo'):
        common.assert_spectral_error(actual, reference, atol=1e-18, bulk_rtol=1e-6)


def test_spatial_assertion_detects_polar_error():
    transform = common.construct_transform(15)
    grid = transform.folded_spec_vod_to_grid_uv(common.zero_vod(15)) + 1
    actual = grid.copy(deep=True)
    actual.u.values[0, 0] += 1
    with pytest.raises(AssertionError, match='polar maximum'):
        common.assert_grid_error(
            actual,
            grid,
            transform.grid.gauss_weights,
            atol=0,
            rms_rtol=1,
            max_rtol=2,
            polar_rtol=0.5,
        )


def test_vector_truncation_without_aliasing(mixed_vod):
    expanded = mixed_vod.pad(l=(0, 8), m=(0, 8), constant_values=0).copy(deep=True)
    expanded.vo.values[:, 0, 12, 3] = 2e-6
    expanded.d.values[:, 1, 15, 15] = 3e-6
    complete = common.construct_transform(15, num_lat=32, num_lon=64)
    truncated = common.construct_transform(7, num_lat=32, num_lon=64)
    winds = complete.folded_spec_vod_to_grid_uv(expanded)
    actual = truncated.grid_uv_to_folded_spec_vod(winds)
    common.assert_spectral_error(actual, mixed_vod, atol=1e-19, bulk_rtol=1e-12)


def test_scalar_projection_is_idempotent(transform):
    generator = np.random.default_rng(42)
    grid = xr.Dataset({
        't': (
            ('latitude', 'longitude'),
            generator.normal(size=(transform.num_lat, transform.num_lon)),
        ),
    })
    projected = transform.grid_to_folded_spec(grid)
    reconstructed = transform.folded_spec_to_grid(projected)
    recovered = transform.grid_to_folded_spec(reconstructed)
    common.assert_spectral_error(recovered, projected, atol=1e-13, bulk_rtol=1e-12)


@pytest.mark.parametrize('name', ['vo', 'd', 'u', 'v'])
def test_vector_missing_variable(name):
    transform = common.construct_transform(7)
    source = common.zero_vod(7)
    if name in {'u', 'v'}:
        source = transform.folded_spec_vod_to_grid_uv(source)
    method = (
        'folded_spec_vod_to_grid_uv'
        if name in {'vo', 'd'}
        else 'grid_uv_to_folded_spec_vod'
    )
    with pytest.raises(AttributeError, match=name):
        getattr(transform, method)(source.drop_vars(name))
