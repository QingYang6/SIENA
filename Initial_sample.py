"""
Set simple co-pol / cross-pol intensity bounds (I1u, I2u) from percentiles of valid pixels.

These bounds feed a direct dark-area mask in Pixel_level_process.thSeg4. They are
not the high / moderate / low probability cuts (those are 0.80 / 0.63 / 0.51).
"""
import numpy as np
from utils import tim


def set_bounds_from_percentiles(ds, args, water_pct_high=95.0, water_pct_low=5.0, valid_only=True):
    """
    Set args.I1u, args.I2u (upper dark-area bounds) and I1d, I2d (lower bounds)
    from percentiles of valid co-pol and cross-pol backscatter.
    """
    lp = np.asarray(ds.img_lp).ravel()
    cp = np.asarray(ds.img_cp).ravel()
    valid = (lp > 0) & (cp > 0) & np.isfinite(lp) & np.isfinite(cp)
    if not np.any(valid):
        # fallback: use small positive defaults
        args.I1u = np.float64(0.01)
        args.I2u = np.float64(0.01)
        args.I1d = np.float64(0.0)
        args.I2d = np.float64(0.0)
        return args

    lp_v = lp[valid]
    cp_v = cp[valid]
    args.I1u = np.float64(np.percentile(lp_v, water_pct_high))
    args.I2u = np.float64(np.percentile(cp_v, water_pct_high))
    args.I1d = np.float64(np.percentile(lp_v, water_pct_low))
    args.I2d = np.float64(np.percentile(cp_v, water_pct_low))
    return args


@tim
def initial_sample(args, ds):
    """Fill I1u / I2u from percentiles when they were not already set."""
    if not hasattr(args, "I1u") or args.I1u is None:
        set_bounds_from_percentiles(ds, args)
    return args, ds
