# ruff: file-ignore[assert, implicit-namespace-package, pytest-parameter-with-default-argument]
import logging
import pathlib

import dask
import dask.callbacks
import numpy as np
import rich.console
import rich.progress
import rich.table
import xarray as xr

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform

logger = logging.getLogger(__name__)


def progress_bar(description):
    columns = (
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn(f'[green]{description}'),
        rich.progress.TaskProgressColumn(),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TextColumn('•'),
        rich.progress.TimeElapsedColumn(),
        rich.progress.TextColumn('•'),
        rich.progress.TimeRemainingColumn(),
    )
    return rich.progress.Progress(*columns, transient=True)


class RichDaskProgressBar(dask.callbacks.Callback):
    def _start(self, _dsk):
        self.progress = progress_bar('dask progress')
        self.progress.start()
        self.task = self.progress.add_task(
            description='the task description',
            total=None,
        )

    def _pretask(self, _key, _dsk, state):
        if state:
            self.update(state)

    def _finish(self, _dsk, state, _err):
        if state:
            self.update(state)
        self.progress.stop()

    def update(self, state):
        num_done = len(state['finished'])
        num_tasks = (
            sum(len(state[k]) for k in ['ready', 'waiting', 'running']) + num_done
        )
        self.progress.update(self.task, total=num_tasks, completed=num_done)


def get_test_data_path():
    return pathlib.Path(__file__).parent / 'new-data'


def open_dataset(filename):
    filename = get_test_data_path() / filename
    logger.info('reading "%s"', filename)
    with xr.open_dataset(filename, engine='h5netcdf') as dataset:
        return dataset.load()


def open_decoded(grid):
    return open_dataset(f'decoded/{grid}.nc').chunk(level=-1)


def open_ds_01_legendre(*, truncation, num_lat, num_lon, **_kwargs):
    return open_dataset(f'test_01_legendre/t{truncation}_{num_lat}_{num_lon}.nc')


def open_ds_05_gradient(*, truncation, num_lat, num_lon, which, **_kwargs):
    return open_dataset(
        f'test_05_gradient/t{truncation}_{num_lat}_{num_lon}_{which}.nc',
    ).chunk(level=-1)


def open_ds_06_wavelets(*, truncation, spline_order, num_splines, **_kwargs):
    return open_dataset(
        f'test_06_wavelets/t{truncation}_{spline_order}_{num_splines}.nc',
    )


def compute_differences(ds_1, ds_2):
    def rms(ds):
        return np.sqrt(np.square(ds).mean())

    variables = [v for v in ds_1 if v in ds_2]
    differences = {}
    for v in variables:
        a_diff = abs(ds_1[v] - ds_2[v])
        mean_a = (abs(ds_1[v]) + abs(ds_2[v])) / 2
        a_max_diff_abs = a_diff.max()
        a_max_diff_rel = xr.where(
            mean_a > 0,
            a_diff / mean_a.where(mean_a > 0, other=1),
            0,
        ).max()
        rms_diff = rms(ds_1[v] - ds_2[v])
        mean_rms = rms(ds_1[v]) + rms(ds_2[v])
        rms_diff_rel = xr.where(
            mean_rms > 0,
            rms_diff / mean_rms.where(mean_rms > 0, other=1),
            0,
        )
        differences[v] = {
            'a_max_diff_abs': a_max_diff_abs,
            'a_max_diff_rel': a_max_diff_rel,
            'rms_diff_abs': rms_diff,
            'rms_diff_rel': rms_diff_rel,
        }
    return dask.compute(differences)[0]


def print_differences(title, errors, rtol, atol):
    def make_string(err, tol):
        the_string = f'{err:.5e}'
        if err < tol or err == 0:
            return the_string
        return '[red]' + the_string + '[/]'

    table = rich.table.Table(title=f'[bold magenta]{title}[/]')
    table.add_column(
        'Variable',
        style='cyan',
    )
    table.add_column('Abs. abs max diff.', justify='right', style='green')
    table.add_column('Rel. abs max diff.', justify='right', style='green')
    table.add_column('Abs. rms diff.', justify='right', style='green')
    table.add_column('Rel. rms diff.', justify='right', style='green')
    for key, value in errors.items():
        table.add_row(
            key,
            make_string(value['a_max_diff_abs'], atol),
            make_string(value['a_max_diff_rel'], rtol),
            make_string(value['rms_diff_abs'], atol),
            make_string(value['rms_diff_rel'], rtol),
        )
    console = rich.console.Console()
    console.print(table)


def test_function(title, ds_out, ds_test, rtol, atol, transform=None):
    if transform is not None:
        np.testing.assert_allclose(
            ds_test.latitude,
            transform.grid.lat,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            ds_test.longitude,
            transform.grid.lon,
            rtol=0,
            atol=1e-12,
        )
        ds_test = ds_test.assign_coords(
            latitude=transform.grid.lat,
            longitude=transform.grid.lon,
        )
    # ensure that progress bar doesn't overwrite test name in pytest
    print()  # ruff: ignore[print]

    # register progress bar
    with RichDaskProgressBar():
        # compute and show differences
        differences = compute_differences(ds_out, ds_test)
        print_differences(
            title=f'Errors on "{title}"',
            errors=differences,
            rtol=rtol,
            atol=atol,
        )

        # check the result against the test data
        xr.testing.assert_allclose(ds_out, ds_test, rtol=rtol, atol=atol)
        if transform is not None:
            floors = {'t': 1e-8, 'vo': 1e-15, 'd': 1e-15, 'u': 1e-9, 'v': 1e-9}
            for name in ds_test:
                assert_grid_spectrum(
                    transform,
                    ds_out[[name]],
                    ds_test[[name]],
                    atol=floors[name],
                    bulk_rtol=rtol,
                )


def construct_transform(truncation, num_lat=None, num_lon=None, dtype='float64'):
    return FullGridSphericalHarmonicsTransform(
        dtype=dtype,
        unfolded_truncation=truncation,
        folded_truncation=truncation,
        num_lat=truncation + 1 if num_lat is None else num_lat,
        num_lon=2 * (truncation + 1) if num_lon is None else num_lon,
        spline_order=None,
        num_splines=None,
        variant=None,
    )


def zero_vod(truncation):
    shape = (2, truncation + 1, truncation + 1)
    return xr.Dataset({
        name: (('c', 'l', 'm'), np.zeros(shape)) for name in ('vo', 'd')
    })


def align_exact(actual, reference):
    assert set(actual.data_vars) == set(reference.data_vars)
    assert dict(actual.sizes) == dict(reference.sizes)
    for name in reference:
        assert set(actual[name].dims) == set(reference[name].dims)
    assert set(actual.coords) == set(reference.coords)
    for name in reference.coords:
        xr.testing.assert_equal(actual[name], reference[name])
    return xr.align(actual, reference, join='exact')


def degree_power(coefficients):
    degree = xr.DataArray(np.arange(coefficients.sizes['l']), dims='l')
    order = xr.DataArray(np.arange(coefficients.sizes['m']), dims='m')
    weights = xr.where(order == 0, 1.0, 2.0)
    return (np.square(coefficients.where(degree >= order, 0)) * weights).sum(
        ('c', 'm'),
        skipna=False,
    )


def assert_error_bound(error, bound, label):
    error, bound = xr.broadcast(error, bound)
    values = np.asarray(error)
    limits = np.asarray(bound)
    assert np.isfinite(values).all(), f'{label}: non-finite error'
    excess = values - limits
    if np.any(excess > 0):
        position = np.unravel_index(np.argmax(excess), values.shape)
        location = {
            dim: error[dim].values[index].item() if dim in error.coords else index
            for dim, index in zip(error.dims, position, strict=True)
        }
        message = (
            f'{label} at {location}: error={values[position]:.6e}, '
            f'bound={limits[position]:.6e}'
        )
        raise AssertionError(message)


def assert_spectral_error(
    actual,
    reference,
    *,
    atol,
    bulk_rtol,
    tail_rtol=None,
    total_rtol=None,
    tail_start=None,
):
    actual, reference = align_exact(actual.compute(), reference.compute())
    tail_rtol = bulk_rtol if tail_rtol is None else tail_rtol
    total_rtol = bulk_rtol if total_rtol is None else total_rtol
    count = reference.sizes['l']
    tail_start = max(1, count * 9 // 10) if tail_start is None else tail_start
    degree = xr.DataArray(np.arange(count), dims='l')
    tolerance = xr.where(degree < tail_start, bulk_rtol, tail_rtol)
    residual = actual - reference
    errors = degree_power(residual)
    powers = degree_power(reference)
    for name in reference:
        worst = abs(residual[name])
        label = f'{name} (worst coefficient={float(worst.max()):.6e})'
        assert_error_bound(
            errors[name],
            atol**2 + tolerance**2 * powers[name],
            f'{label}, degree',
        )
        for band, selection, rtol in (
            ('bulk', slice(0, tail_start), bulk_rtol),
            ('tail', slice(tail_start, count), tail_rtol),
            ('total', slice(0, count), total_rtol),
        ):
            error = errors[name].isel(l=selection)
            power = powers[name].isel(l=selection)
            assert_error_bound(
                error.sum('l'),
                error.sizes['l'] * atol**2 + rtol**2 * power.sum('l'),
                f'{label}, {band}',
            )


def assert_grid_error(
    actual,
    reference,
    gauss_weights,
    *,
    atol,
    rms_rtol,
    max_rtol,
    polar_rtol,
):
    actual, reference = align_exact(actual.compute(), reference.compute())
    weights = xr.DataArray(
        gauss_weights,
        dims='latitude',
        coords={'latitude': reference.latitude},
    )
    for name in reference:
        residual = actual[name] - reference[name]
        reference_rms = np.sqrt(
            np
            .square(reference[name])
            .weighted(weights)
            .mean(('latitude', 'longitude')),
        )
        rms = np.sqrt(
            np.square(residual).weighted(weights).mean(('latitude', 'longitude')),
        )
        assert_error_bound(rms, atol + rms_rtol * reference_rms, f'{name}, grid RMS')
        assert_error_bound(
            abs(residual).max(('latitude', 'longitude')),
            atol + max_rtol * reference_rms,
            f'{name}, grid maximum',
        )
        polar = abs(residual).where(abs(reference.latitude) >= 75, drop=True)
        if polar.sizes['latitude']:
            assert_error_bound(
                polar.max(('latitude', 'longitude')),
                atol + polar_rtol * reference_rms,
                f'{name}, polar maximum',
            )


def assert_grid_spectrum(transform, actual, reference, **budget):
    actual, reference = align_exact(actual, reference)
    assert_spectral_error(
        transform.grid_to_folded_spec(actual),
        transform.grid_to_folded_spec(reference),
        **budget,
    )
