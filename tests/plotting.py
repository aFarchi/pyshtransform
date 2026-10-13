# ruff: file-ignore[implicit-namespace-package]
import warnings

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def select_field(dataset, var, level):
    field = dataset[var]
    if 'level' in field.dims:
        if level is None:
            message = 'Select a level before plotting a multi-level field'
            raise ValueError(message)
        field = field.sel(level=level)
    return field.compute()


def make_map(
    ds_ref,
    ds_new,
    var,
    level=None,
    vmax=None,
    *,
    projection=None,
    coastlines=False,
):
    reference = select_field(ds_ref, var, level).transpose('latitude', 'longitude')
    computed = select_field(ds_new, var, level).transpose('latitude', 'longitude')
    reference, computed = xr.align(reference, computed, join='exact')
    residual = computed - reference
    field_min = min(float(reference.min()), float(computed.min()))
    field_max = max(float(reference.max()), float(computed.max()))
    if vmax is not None:
        field_min, field_max = -vmax, vmax
    residual_max = max(float(abs(residual).max()), 1e-15)
    subplot_kwargs = {}
    mesh_kwargs = {}
    if projection is not None:
        subplot_kwargs['projection'] = projection
        mesh_kwargs['transform'] = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
        layout='constrained',
        subplot_kw=subplot_kwargs,
    )
    meshes = []
    for axis, field, title in zip(
        axes,
        (reference, computed, residual),
        ('Reference', 'Computed', 'Computed - reference'),
        strict=True,
    ):
        is_residual = title == 'Computed - reference'
        meshes.append(
            axis.pcolormesh(
                field.longitude,
                field.latitude,
                field,
                shading='auto',
                cmap='RdBu_r' if is_residual else 'viridis',
                vmin=-residual_max if is_residual else field_min,
                vmax=residual_max if is_residual else field_max,
                **mesh_kwargs,
            ),
        )
        axis.set_title(title)
        if projection is not None:
            if coastlines:
                axis.coastlines()
            axis.set_global()
        else:
            axis.set_xlabel('Longitude (degrees)')
            axis.set_ylabel('Latitude (degrees)')
    unit = reference.attrs.get('units', 'field units')
    figure.colorbar(
        meshes[0],
        ax=axes[:2],
        label=unit,
        shrink=0.8,
        orientation='horizontal',
        location='bottom',
        pad=0.12,
    )
    figure.colorbar(
        meshes[2],
        ax=axes[2],
        label=f'Error ({unit})',
        shrink=0.8,
        orientation='horizontal',
        location='bottom',
        pad=0.12,
    )
    suffix = '' if level is None else f', level {level}'
    figure.suptitle(f'{var}{suffix}')
    return figure


def show_map(
    ds_ref,
    ds_new,
    var,
    level=None,
    vmax=None,
    *,
    projection=None,
    coastlines=False,
    title=None,
    panel_titles=None,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'invalid value encountered in create_collection$',
            category=RuntimeWarning,
            module=r'shapely\.creation$',
        )
        figure = make_map(
            ds_ref,
            ds_new,
            var,
            level,
            vmax,
            projection=projection,
            coastlines=coastlines,
        )
        if title is not None:
            figure.suptitle(title)
        if panel_titles is not None:
            for axis, panel_title in zip(figure.axes[:3], panel_titles, strict=True):
                axis.set_title(panel_title)
        plt.show()
    return figure


def make_ps_plot(ds_ref, ds_new, ds_diff, var, level=None):
    reference = select_field(ds_ref, var, level)
    computed = select_field(ds_new, var, level)
    residual = select_field(ds_diff, var, level)
    reference, computed, residual = xr.align(
        reference,
        computed,
        residual,
        join='exact',
    )
    degree = (
        reference.l.values
        if 'l' in reference.coords
        else np.arange(reference.sizes['l'])
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for power, label in (
        (reference, 'Reference'),
        (computed, 'Computed'),
        (residual, 'Residual'),
    ):
        axes[0].plot(degree, power.where(power > 0), label=label)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Power')
    axes[0].set_title('Degree-wise spectra')
    axes[0].legend()
    relative = np.sqrt(residual / reference.where(reference > 0))
    axes[1].plot(degree, relative.where(relative > 0), color='tab:red')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('sqrt(residual power / reference power)')
    axes[1].set_title('Relative residual amplitude (diagnostic)')
    for axis in axes:
        axis.set_xlabel('Spherical-harmonic degree')
        axis.set_xlim(0, max(1, int(degree[-1])))
        axis.grid(visible=True, alpha=0.3)
    suffix = '' if level is None else f', level {level}'
    figure.suptitle(f'{var}{suffix}; zero power omitted on logarithmic axes')
    return figure
