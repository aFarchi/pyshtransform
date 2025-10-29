import common
import pytest

from pyshtransform.full_grid.transformation import FullGridSphericalHarmonicsTransform


@pytest.fixture(
    params=[
        dict(
            truncation=15,
            num_lat=64,
            num_lon=128,
        ),
        dict(
            truncation=31,
            num_lat=32,
            num_lon=64,
        ),
        dict(
            truncation=63,
            num_lat=64,
            num_lon=127,
        ),
    ]
)
def config(request):
    return request.param


@pytest.fixture
def ds_05_gradient_t(config):
    return common.open_ds_05_gradient(which='t', **config)


@pytest.fixture
def ds_05_gradient_dt_dtheta(config):
    return common.open_ds_05_gradient(which='dt_dtheta', **config)


@pytest.fixture
def ds_05_gradient_dt_dphi(config):
    return common.open_ds_05_gradient(which='dt_dphi', **config)


def construct_transformation(ds_05_gradient_t):
    return FullGridSphericalHarmonicsTransform(
        dtype='float64',
        unfolded_truncation=1279,
        folded_truncation=ds_05_gradient_t.truncation,
        num_lat=len(ds_05_gradient_t.latitude),
        num_lon=len(ds_05_gradient_t.longitude),
        spline_order=None,
        num_splines=None,
        variant='',
    )


def test_forward(ds_spec, ds_05_gradient_t):
    transformation = construct_transformation(ds_05_gradient_t)
    ds_grid = transformation.unfolded_spec_to_grid(ds_spec)
    common.test_function(
        'spec_to_grid',
        ds_grid,
        ds_05_gradient_t,
        rtol=1e-7,
        atol=0,
    )


def test_grad_theta(ds_spec, ds_05_gradient_dt_dtheta):
    transformation = construct_transformation(ds_05_gradient_dt_dtheta)
    ds_grid = transformation.unfolded_spec_to_grid_grad_theta(ds_spec, prefix='')
    common.test_function(
        'spec_to_grid_grad_theta',
        ds_grid,
        ds_05_gradient_dt_dtheta,
        rtol=1e-7,
        atol=0,
    )


def test_grad_phi(ds_spec, ds_05_gradient_dt_dphi):
    transformation = construct_transformation(ds_05_gradient_dt_dphi)
    ds_grid = transformation.unfolded_spec_to_grid_grad_phi(ds_spec, prefix='')
    common.test_function(
        'spec_to_grid_grad_phi',
        ds_grid,
        ds_05_gradient_dt_dphi,
        rtol=1e-7,
        atol=0,
    )


def test_full(
    ds_spec, ds_05_gradient_t, ds_05_gradient_dt_dtheta, ds_05_gradient_dt_dphi
):
    transformation = construct_transformation(ds_05_gradient_dt_dphi)
    ds_grid = transformation.unfolded_spec_to_grid_full(ds_spec)
    ds_grid_selected = ds_grid.drop_vars(
        (var for var in ds_grid if var.startswith('gt') or var.startswith('gp'))
    )
    common.test_function(
        'spec_to_grid_full_regular',
        ds_grid_selected,
        ds_05_gradient_t,
        rtol=1e-7,
        atol=0,
    )
    ds_grid_selected = ds_grid.drop_vars(
        (var for var in ds_grid if not var.startswith('gt'))
    )
    ds_grid_selected = ds_grid_selected.rename({
        var: var.replace('gt', '') for var in ds_grid_selected
    })
    common.test_function(
        'spec_to_grid_full_grad_theta',
        ds_grid_selected,
        ds_05_gradient_dt_dtheta,
        rtol=1e-7,
        atol=0,
    )
    ds_grid_selected = ds_grid.drop_vars(
        (var for var in ds_grid if not var.startswith('gp'))
    )
    ds_grid_selected = ds_grid_selected.rename({
        var: var.replace('gp', '') for var in ds_grid_selected
    })
    common.test_function(
        'spec_to_grid_full_grad_phi',
        ds_grid_selected,
        ds_05_gradient_dt_dphi,
        rtol=1e-7,
        atol=0,
    )
