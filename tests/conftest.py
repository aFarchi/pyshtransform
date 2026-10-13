import common  # ruff: ignore[implicit-namespace-package]
import pytest


@pytest.fixture
def ds_spec():
    return common.open_decoded('t255')


@pytest.fixture(params=['f32', 'f96'])
def ds_grid_ref(request):
    return common.open_decoded(request.param)
