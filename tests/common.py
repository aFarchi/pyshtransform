import logging
import pathlib

import dask
import dask.callbacks
import numpy as np
import rich.console
import rich.progress
import rich.table
import xarray as xr

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
            description='the task description', total=None
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
    return pathlib.Path(__file__).parent / 'data'


def open_dataset(filename):
    filename = get_test_data_path() / filename
    logger.info(f'reading "{filename}"')
    return xr.open_dataset(filename, engine='h5netcdf')


def open_decoded(grid):
    return open_dataset(f'decoded/{grid}.nc').chunk(level=-1)


def open_ds_01_legendre(*, truncation, num_lat, num_lon, **_kwargs):
    return open_dataset(f'test_01_legendre/t{truncation}_{num_lat}_{num_lon}.nc')


def open_ds_05_gradient(*, truncation, num_lat, num_lon, which, **_kwargs):
    return open_dataset(
        f'test_05_gradient/t{truncation}_{num_lat}_{num_lon}_{which}.nc'
    ).chunk(level=-1)


def open_ds_06_wavelets(*, truncation, spline_order, num_splines, **_kwargs):
    return open_dataset(
        f'test_06_wavelets/t{truncation}_{spline_order}_{num_splines}.nc'
    )


def compute_differences(ds_1, ds_2):
    def rms(ds):
        return np.sqrt(np.square(ds).mean())

    variables = [v for v in ds_1 if v in ds_2]
    differences = dict()
    for v in variables:
        a_diff = abs(ds_1[v] - ds_2[v])
        mean_a = (abs(ds_1[v]) + abs(ds_2[v])) / 2
        a_max_diff_abs = a_diff.max()
        a_max_diff_rel = xr.where(
            mean_a > 0,
            a_diff / mean_a,
            0,
        ).max()
        rms_diff = rms(ds_1[v] - ds_2[v])
        mean_rms = rms(ds_1[v]) + rms(ds_2[v])
        rms_diff_rel = xr.where(
            mean_rms > 0,
            rms_diff / mean_rms,
            0,
        )
        differences[v] = dict(
            a_max_diff_abs=a_max_diff_abs,
            a_max_diff_rel=a_max_diff_rel,
            rms_diff_abs=rms_diff,
            rms_diff_rel=rms_diff_rel,
        )
    return dask.compute(differences)[0]


def print_differences(title, errors, rtol, atol):
    def make_string(err, tol):
        the_string = f'{err:.5e}'
        if err < tol or err == 0:
            return the_string
        else:
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


def test_function(title, ds_out, ds_test, rtol, atol):
    # ensure that progress bar doesn't overwrite test name in pytest
    print()

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
