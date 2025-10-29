"""Implementation of a padding transformation in spectral space."""

import xarray as xr

from pyshtransform.folding.transformation import FoldingTransformation


class PaddingTransformation(FoldingTransformation):
    """Padding transformation in spectral space."""

    def __init__(self, dtype: str, truncation: int):
        """Initialises the padding transformation.

        Args:
            dtype: Output floating-point data type.
            truncation: Output truncation.
        """
        super().__init__(dtype=dtype, truncation=truncation, factor=1)

    def apply(self, ds_data: xr.Dataset) -> xr.Dataset:
        """Applies the padding transformation.

        Args:
            ds_data: Dataset in folded spectral space.

        Returns:
            Padded dataset in unfolded spectral space.
        """
        padding = self.truncation + 1 - len(ds_data.l)
        ds_data = ds_data.pad(l=(0, padding), m=(0, padding), constant_values=0)
        ds_data = ds_data.chunk(l=-1, m=-1)
        return self.unfold_clm(ds_data)
