import common
import pytest


@pytest.fixture
def ds_spec():
    return common.open_decoded('sh')


@pytest.fixture(params=['f32', 'f96'])
def ds_grid_ref(request):
    return common.open_decoded(request.param)
