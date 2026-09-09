"""
Pixel-level detection: score each SAR pixel as water-like, then cut three masks.

Flow: Initial_sample (intensity bounds) → run_fuzzy_core (water likelihood) →
thSeg4 (high / moderate / low probability water masks).

The three cuts on the normalized fuzzy score (edit in thSeg4 if you want to tune):
  high-probability water      > 0.80
  moderate-probability water  > 0.63
  low-probability water       > 0.51
These are nested: high ⊂ moderate ⊂ low. Object-level rules later decide which
candidate water bodies to keep. See SIENA.py and README.md.
"""
import os
import numpy as np
import rasterio
from scipy import ndimage

from utils import tim, write_raster, create_raster_data_dict
from Initial_sample import initial_sample
from Fuzzy_logic_core import run_fuzzy_core


def normalize_water_score(water_score, intensity_mask, use_percentiles=False, lower_pct=5.0, upper_pct=95.0):
    """
    Stretch the fuzzy water score to [0, 1] so the three probability cuts can be applied.

    Default is min–max over valid pixels. Set use_percentiles=True (or env
    SIENA_NORMALIZE_PERCENTILE=1) to use lower_pct/upper_pct instead, which
    reduces the effect of a few extreme pixels on the scale.
    """
    water_score_masked = np.where(intensity_mask > 0, water_score, np.nan)
    if use_percentiles:
        min_val = np.nanpercentile(water_score_masked, lower_pct)
        max_val = np.nanpercentile(water_score_masked, upper_pct)
    else:
        min_val = np.nanmin(water_score_masked)
        max_val = np.nanmax(water_score_masked)
    if max_val > min_val:
        normalized_score = (water_score_masked - min_val) / (max_val - min_val)
    else:
        normalized_score = np.where(~np.isnan(water_score_masked), 1, np.nan)
    return normalized_score


def filter_small_pixel_groups(High_mask_np):
    """Drop tiny connected groups on the high-probability mask (speckle). Size grows slowly with image width/height."""
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    height, width = High_mask_np.shape
    longest_axis = max(height, width)
    size_threshold = 1 + (longest_axis // 4000)
    print("Adaptive size threshold:", size_threshold)
    labeled_flood_mask, num_labels = ndimage.label(High_mask_np, structure=structure)
    label_sizes = ndimage.sum(High_mask_np, labeled_flood_mask, range(num_labels + 1))
    small_labels = np.where(label_sizes < size_threshold)[0]
    labeled_flood_mask[np.isin(labeled_flood_mask, small_labels)] = 0
    return np.where(labeled_flood_mask > 0, 1, 0).astype(np.uint8)


def generate_threshold_masks(ds, normalized_water_scores, high_threshold, mod_threshold, low_threshold, mask_profile, args):
    """
    Build high-, moderate-, and low-probability water masks from the normalized score.
    The same three cuts are used for all land-cover types. Tiny speckle is removed
    from the high-probability mask only.
    """
    valid = np.isfinite(normalized_water_scores) & (np.asarray(ds.img_lp) > 0)
    High_mask = np.where(valid & (normalized_water_scores > high_threshold), 1, 0).astype(np.uint8)
    Mod_mask = np.where(valid & (normalized_water_scores > mod_threshold), 1, 0).astype(np.uint8)
    Low_mask = np.where(valid & (normalized_water_scores > low_threshold), 1, 0).astype(np.uint8)
    High_mask_filtered = filter_small_pixel_groups(High_mask.astype(np.uint8))
    I_pol1 = getattr(ds, "I_pol1", ds.img_lp)
    if getattr(args, "output_intermediate", False):
        save_mask_to_file(High_mask_filtered, I_pol1, mask_profile, args.fileOutHigh)
        save_mask_to_file(Mod_mask, I_pol1, mask_profile, args.fileOutMid)
        save_mask_to_file(Low_mask, I_pol1, mask_profile, args.fileOutLow)
        print("Masks written: high=%s, moderate=%s, low=%s" % (args.fileOutHigh, args.fileOutMid, args.fileOutLow))
    return High_mask_filtered, Mod_mask.astype(np.uint8), Low_mask.astype(np.uint8)


def save_mask_to_file(mask, reference_band, mask_profile, output_path):
    """Write one binary mask GeoTIFF; no-data where the SAR band is invalid."""
    mask_write = np.where(reference_band > 0, np.asarray(mask), mask_profile["nodata"]).astype(mask_profile["dtype"])
    with rasterio.open(output_path, "w", **mask_profile, compress="deflate") as dst:
        dst.write(mask_write, indexes=1)


@tim
def thSeg4(args, ds):
    """
    Cut the normalized fuzzy score into high / moderate / low probability water.

    Also builds a simple direct_mask from co-pol and cross-pol intensity bounds
    (I1u, I2u) used later for optional display products.
    """
    args.fileOutHigh = os.path.join(args.dirOut, "waterMask_highProb.tif")
    args.fileOutMid = os.path.join(args.dirOut, "waterMask_modProb.tif")
    args.fileOutLow = os.path.join(args.dirOut, "waterMask_lowProb.tif")
    args.bcoutname = ["High_mask", "Mod_mask", "Low_mask"]
    direct_mask_raw = np.where(np.logical_and(ds.img_lp <= args.I1u, ds.img_cp <= args.I2u), 1, 0).astype(np.uint8)
    ds.direct_mask = np.where(ds.img_lp > 0, direct_mask_raw, 0).astype(np.uint8)
    mask_profile = args.refmeta.copy()
    mask_profile.update(dtype=rasterio.uint8, nodata=255)
    # Set True to use 5–95th percentile instead of min-max (reduces impact of scene-wide stat shift)
    use_percentile_norm = os.environ.get("SIENA_NORMALIZE_PERCENTILE", "0").strip().lower() in ("1", "true", "yes")
    normalized_water_scores = normalize_water_score(
        ds.fuzzy_scores["fuzzy_score"], ds.img_lp,
        use_percentiles=use_percentile_norm, lower_pct=5.0, upper_pct=95.0
    )
    ds.normalized_water_scores = normalized_water_scores
    # Nested water-probability cuts on the normalized fuzzy score (tune here).
    # High = strongest seeds; moderate = intermediate; low = broadest candidates.
    high_threshold, mod_threshold, low_threshold = 0.80, 0.63, 0.51
    print("Generating three probability masks (high > %.2f, moderate > %.2f, low > %.2f)..." % (
        high_threshold, mod_threshold, low_threshold))
    High_mask, Mod_mask, Low_mask = generate_threshold_masks(
        ds, normalized_water_scores, high_threshold, mod_threshold, low_threshold, mask_profile, args
    )
    setattr(ds, args.bcoutname[0], High_mask)
    setattr(ds, args.bcoutname[1], Mod_mask)
    setattr(ds, args.bcoutname[2], Low_mask)
    return args, ds


@tim
def run_pixel_level_process(args, ds):
    """
    Pixel-level detection for one scene.

    Sets intensity bounds, computes a per-pixel water likelihood, then writes
    ds.High_mask, ds.Mod_mask, ds.Low_mask (high / moderate / low probability
    water) and ds.normalized_water_scores for Object_level_process.
    """
    args, ds = initial_sample(args, ds)
    fuzzy_results = run_fuzzy_core(ds)
    ds.fuzzy_scores = fuzzy_results
    ds.I_pol1 = np.where(ds.img_lp > 0, ds.img_lp, np.nan)
    ds.I_pol2 = np.where(ds.img_cp > 0, ds.img_cp, np.nan)
    fuzzy_arr = np.where(np.isfinite(ds.fuzzy_scores["fuzzy_score"]), ds.fuzzy_scores["fuzzy_score"], np.nan)
    fuzzy_d_m = create_raster_data_dict(fuzzy_arr.astype(np.float32), args.refmeta.copy(), None)
    args.fileOutFuzzy = os.path.join(args.dirOut, "Fuzzy_score.tif")
    if getattr(args, "output_intermediate", False):
        write_raster(fuzzy_d_m, args.fileOutFuzzy)
    args, ds = thSeg4(args, ds)
    return args, ds
