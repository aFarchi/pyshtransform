import common

from pyshtransform.folding.transformation import FoldingTransformation


def test_folding(ds_spec):
    t_1279_3 = FoldingTransformation(
        dtype='float64',
        unfolded_truncation=1279,
        folded_truncation=3,
        factor=1,
        variant=None,
    )
    t_1279_5 = FoldingTransformation(
        dtype='float64',
        unfolded_truncation=1279,
        folded_truncation=5,
        factor=1,
        variant=None,
    )
    t_3_3 = FoldingTransformation(
        dtype='float64',
        unfolded_truncation=3,
        folded_truncation=3,
        factor=1,
        variant=None,
    )
    t_5_5 = FoldingTransformation(
        dtype='float64',
        unfolded_truncation=5,
        folded_truncation=5,
        factor=1,
        variant=None,
    )
    t_3_5 = FoldingTransformation(
        dtype='float64',
        unfolded_truncation=3,
        folded_truncation=5,
        factor=1,
        variant=None,
    )
    t_5_3 = FoldingTransformation(
        dtype='float64',
        unfolded_truncation=5,
        folded_truncation=3,
        factor=1,
        variant=None,
    )
    ds_folded_3 = t_1279_3.fold_clm(ds_spec)
    ds_folded_5 = t_1279_5.fold_clm(ds_spec)
    ds_unfolded_3 = t_3_3.unfold_clm(ds_folded_3)
    ds_unfolded_5 = t_5_5.unfold_clm(ds_folded_5)
    ds_folded_3_padded = ds_folded_3.pad(l=(0, 2), m=(0, 2), constant_values=0).chunk(
        l=-1, m=-1
    )
    ds_unfolded_3_padded = t_5_5.unfold_clm(ds_folded_3_padded)
    common.test_function(
        'folding with padding',
        t_3_5.fold_clm(ds_unfolded_3),
        ds_folded_3_padded,
        rtol=0,
        atol=0,
    )
    common.test_function(
        'folding with truncation',
        t_5_3.fold_clm(ds_unfolded_5),
        ds_folded_3,
        rtol=0,
        atol=0,
    )
    common.test_function(
        'unfolding with truncation',
        t_3_5.unfold_clm(ds_folded_5),
        ds_unfolded_3,
        rtol=0,
        atol=0,
    )
    common.test_function(
        'unfolding with padding',
        t_5_3.unfold_clm(ds_folded_3),
        ds_unfolded_3_padded,
        rtol=0,
        atol=0,
    )
