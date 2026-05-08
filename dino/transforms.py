import math
from warpconvnet.geometry.types.voxels import Voxels


class FeatureLogTransform:
    """
    Maps raw charge [ADC] to [-1, +1] via log10 compression, in-place:

        y = 2 * (log10(x + min_val) - log10(min_val)) / (log10(max_val + min_val) - log10(min_val)) - 1
    
    x = 0 becomes y = -1  (never preset in sparse; yet everthing below min_val pushed to -1)
    x = max_val becomes y = +1 (but there is no clipping, so values above max_val can be > +1)

    min_val and max_val should be set from the dataset charge distribution
    (e.g. 2nd and 99.999th percentile) and stored in DINOConfig.
    
    See explore_normalization.ipynb for more details and visualization of the transform.
    """

    def __init__(self, min_val: float, max_val: float) -> None:
        self.min_val = min_val
        y0 = math.log10(min_val)
        y1 = math.log10(max_val + min_val)
        self._y0    = y0
        self._scale = 2.0 / (y1 - y0)

    def __call__(self, xs: Voxels) -> Voxels:
        feats = xs.feature_tensor
        feats.add_(self.min_val).log10_().sub_(self._y0).mul_(self._scale).add_(-1.0)
        return xs
