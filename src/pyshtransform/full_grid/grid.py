"""Full Gaussian grid."""

import numpy as np

from pyshtransform.legendre import gauss_legendre_nodes, plmbar_d1


class FullGrid:
    """Full Gaussian grid.

    Attributes:
        lat: Latitude nodes.
        lon: Longitude nodes.
        plm: Plm coefficients for the spectral transformation.
        alm: Alm coefficients for the spectral transformation.
        pw: Gauss--Legendre weights for the spectral transformation.
    """

    def __init__(
        self,
        dtype: str,
        truncation: int,
        num_lat: int,
        num_lon: int,
    ):
        cos_t, w = gauss_legendre_nodes(num_lat)
        self.lat = np.asin(cos_t) * 180 / np.pi
        self.lon = np.linspace(0, 360, num_lon, endpoint=False)
        cos_l = np.cos(self.lat * np.pi / 180)
        p, a = plmbar_d1(truncation, cos_t)
        self.plm = p.astype(dtype)
        self.alm = (a * np.expand_dims(cos_l, (1, 2))).astype(dtype)
        self.pw = (0.5 * p * np.expand_dims(w, (1, 2))).astype(dtype)
