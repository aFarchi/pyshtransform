from pyshtransform.folding.transformation import FoldingTransformation


class PaddingTransformation(FoldingTransformation):
    def __init__(self, dtype, truncation):
        super().__init__(dtype=dtype, truncation=truncation, factor=1)

    def apply(self, ds_data):
        padding = self.truncation + 1 - len(ds_data.l)
        ds_data = ds_data.pad(l=(0, padding), m=(0, padding), constant_values=0)
        ds_data = ds_data.chunk(l=-1, m=-1)
        return self.unfold_clm(ds_data)
