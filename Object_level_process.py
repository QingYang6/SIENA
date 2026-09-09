"""
Object-level refinement: include or exclude candidate water bodies with rules.

Pixel-level detection produces high / moderate / low probability water masks.
This module groups those pixels into objects and applies hydrologic and
land-cover tests (urban fraction, HAND, slope, desert/snow, high–low overlap,
bimodality of SAR intensity). Rule ratios are CLI flags on SIENA.py.

Default path: keep low-probability objects that intersect a high-probability
object, then preserve / remove / check remaining flood-water bodies.
Optional --use_region_growth grows from high seeds into the low-score neighborhood.

See SIENA.py and README.md for the tunable ratios.
"""
import math
import os
import time
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.optimize import curve_fit
from utils import tim
from Fuzzy_logic_core import compute_slope_dem


def _pixel_size_m(refmeta):
    """Approximate pixel size in meters from refmeta (transform + optional crs)."""
    t = refmeta.get("transform")
    if t is None:
        return None
    w, h = refmeta.get("width"), refmeta.get("height")
    # Center pixel in image coords
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    lon_c, lat_c = t * (cx, cy)
    dx_deg, dy_deg = abs(t.a), abs(t.e)
    crs = refmeta.get("crs")
    is_geo = getattr(crs, "is_geographic", False) if crs is not None else False
    if callable(is_geo):
        is_geo = is_geo()
    if crs is not None and is_geo:
        m_per_deg_x = 111320 * 1000 * math.cos(math.radians(lat_c))
        m_per_deg_y = 111320 * 1000
        return (dx_deg * m_per_deg_x + dy_deg * m_per_deg_y) / 2.0
    # Projected CRS: assume units are meters
    return (abs(t.a) + abs(t.e)) / 2.0


# Grow water from high-probability seeds into neighboring lower-score pixels.

def region_grow_from_fuzzy(
    fuzzy_norm,
    high_threshold=0.81,
    relaxed_threshold=0.51,
    valid_mask=None,
    initial_binary=None,
    maxiter=10000,
):
    """
    Grow water from high-probability seeds into adjacent pixels whose fuzzy score
    is still above relaxed_threshold. Used only when --use_region_growth is set.
    fuzzy_norm: 2D float [0, 1]. valid_mask: optional 2D bool (growth only inside valid).
    initial_binary: optional 2D 0/1 seed; if omitted, seeds are fuzzy_norm > high_threshold.
    Returns binary 2D uint8 (1 = water).
    """
    if high_threshold <= relaxed_threshold:
        raise ValueError("high_threshold must be > relaxed_threshold")
    if initial_binary is not None:
        binary = np.asarray(initial_binary, dtype=np.uint8)
        if binary.max() > 1:
            binary = np.where(binary > 0, 1, 0).astype(np.uint8)
    else:
        binary = (fuzzy_norm > high_threshold).astype(np.uint8)
    if valid_mask is not None:
        binary = np.where(valid_mask, binary, 0)
    itercount = 0
    while itercount < maxiter:
        dilated = ndimage.binary_dilation(binary.astype(bool))
        if valid_mask is not None:
            dilated = dilated & valid_mask
        buffer_binary = dilated & (~binary.astype(bool))
        add_cand = (fuzzy_norm > relaxed_threshold) & buffer_binary
        n_add = np.sum(add_cand)
        if n_add == 0:
            break
        binary = np.where(add_cand, 1, binary).astype(np.uint8)
        itercount += 1
    return binary


# --- Object-level include / exclude rules (groupby per labeled water body) ---

def _remove_objects_by_flags(boolean_list, group_series, labeled_mask, CM):
    """Zero out labeled objects whose flags say they should be excluded."""
    remove_list = group_series.index[boolean_list].tolist()
    mask = np.isin(labeled_mask, remove_list)
    return np.where(mask, 0, CM).astype(np.uint8)


def apply_high_mask_desert_rule(mask, land_cover, bare_class=60, snow_ice_class=70, bare_ratio_thresh=0.7):
    """
    Drop objects whose bare/sparse or snow/ice fraction is at least bare_ratio_thresh.
    Used separately on high, moderate, and low probability masks (different ratios).
    ESA WorldCover: 60 = bare / sparse vegetation, 70 = snow and ice.
    """
    labeled, num_labels = ndimage.label(mask.astype(np.uint8), structure=ndimage.generate_binary_structure(2, 2))
    if num_labels == 0:
        return mask.astype(np.uint8)
    LCC = np.asarray(land_cover)
    is_water = np.where(labeled > 0, 1, 0).astype(np.uint8)
    is_bare = np.where(LCC == bare_class, 1, 0).astype(np.uint8)
    is_snow_ice = np.where(LCC == snow_ice_class, 1, 0).astype(np.uint8)
    CM = labeled.astype(np.int64)
    df = pd.DataFrame({
        "CM": CM.ravel(),
        "isWater": is_water.ravel(),
        "bare": is_bare.ravel(),
        "snow_ice": is_snow_ice.ravel(),
    })
    df = df.loc[df.CM != 0]
    grouped = df.groupby("CM")
    objSum = grouped.isWater.sum()
    obj_rBare = grouped.bare.sum() / objSum
    obj_rSnowIce = grouped.snow_ice.sum() / objSum
    boolean_list = (obj_rBare >= bare_ratio_thresh) | (obj_rSnowIce >= bare_ratio_thresh)
    morph_mask_raw = _remove_objects_by_flags(boolean_list, objSum, labeled, CM)
    return np.where(morph_mask_raw > 0, 1, 0).astype(np.uint8)


def apply_high_mask_wo_nearby_rule(
    high_mask,
    water_occurrence,
    pixel_size_m,
    buffer_m=500.0,
    wo_threshold=5.0,
    min_ratio=0.01,
):
    """
    Optional rule (not on the default SIENA.py path): drop a high-probability object
    unless nearby water-occurrence pixels (WO > wo_threshold within buffer_m) cover
    at least min_ratio of the object's size. CLI: --object_wo_threshold, --object_wo_min_ratio.
    """
    high_mask = np.asarray(high_mask, dtype=np.uint8)
    wo = np.asarray(water_occurrence, dtype=np.float64)
    if pixel_size_m is None or pixel_size_m <= 0:
        return high_mask
    buffer_pixels = max(1, int(round(buffer_m / pixel_size_m)))
    # Disk structure for "within buffer_m" (Euclidean-ish)
    r = buffer_pixels
    yx = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (yx[0] * yx[0] + yx[1] * yx[1] <= r * r).astype(np.uint8)
    wo_binary = (wo > wo_threshold).astype(np.uint8)
    valid_wo = np.isfinite(wo)
    wo_binary = np.where(valid_wo, wo_binary, 0)

    labeled, num_labels = ndimage.label(high_mask, structure=ndimage.generate_binary_structure(2, 2))
    if num_labels == 0:
        return high_mask
    out = np.asarray(high_mask, dtype=np.uint8)
    # Use bounding box per object so we only dilate a small crop (much faster for many objects)
    slices = ndimage.find_objects(labeled)
    ny, nx = labeled.shape
    margin = min(buffer_pixels + 2, max(ny, nx))  # clip margin to image size
    for lab in range(1, num_labels + 1):
        sl = slices[lab - 1]
        if sl is None:
            continue
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        # Crop with margin for dilation
        sr0 = max(0, r0 - margin)
        sr1 = min(ny, r1 + margin)
        sc0 = max(0, c0 - margin)
        sc1 = min(nx, c1 + margin)
        obj_crop = (labeled[sr0:sr1, sc0:sc1] == lab)
        obj_size = int(np.sum(obj_crop))
        if obj_size == 0:
            continue
        wo_crop = wo_binary[sr0:sr1, sc0:sc1]
        dilated = ndimage.binary_dilation(obj_crop, structure=disk)
        wo_in_buffer = int(np.sum(wo_crop & dilated))
        if wo_in_buffer < min_ratio * obj_size:
            out[labeled == lab] = 0
    return out


def apply_object_rules(
    mask,
    land_cover,
    nWBClasses,
    rNWB_WST_Org=0.05,
    min_pixels=3,
    hand=None,
    hand_max=None,
    slope=None,
    slope_max=None,
):
    """
    Drop objects that fail size, urban-fraction, HAND, or slope tests.

    - Urban: fraction of pixels in nWBClasses (default built-up = 50) >= rNWB_WST_Org
      (CLI default 0.3 = 30%).
    - Min size: pixel count < min_pixels (speckle).
    - HAND: if provided, drop if median HAND (m) > hand_max (--object_hand_max_m).
    - Slope: if provided, drop if median slope (deg) > slope_max (--object_slope_max_deg).
    """
    labeled, num_labels = ndimage.label(mask.astype(np.uint8), structure=ndimage.generate_binary_structure(2, 2))
    if num_labels == 0:
        return mask.astype(np.uint8)
    LCC = np.asarray(land_cover)
    is_water = np.where(labeled > 0, 1, 0).astype(np.uint8)
    rasurbanLCC_value = np.where(np.isin(LCC, nWBClasses), 1, 0).astype(np.uint8)
    CM = labeled.astype(np.int64)
    df = pd.DataFrame({
        "CM": CM.ravel(),
        "isWater": is_water.ravel(),
        "LCC": rasurbanLCC_value.ravel(),
    })
    if hand is not None:
        df["hand"] = np.asarray(hand, dtype=np.float64).ravel()
    if slope is not None:
        df["slope"] = np.asarray(slope, dtype=np.float64).ravel()
    df = df.loc[df.CM != 0]
    grouped = df.groupby("CM")
    objSum = grouped.isWater.sum()
    objSumNWB = grouped.LCC.sum()
    obj_rNWB = objSumNWB / objSum
    boolean_list = (objSum < min_pixels) | (obj_rNWB >= rNWB_WST_Org)
    if hand is not None and hand_max is not None:
        obj_median_hand = grouped["hand"].median()
        boolean_list = boolean_list | (obj_median_hand > hand_max)
    if slope is not None and slope_max is not None:
        obj_median_slope = grouped["slope"].median()
        boolean_list = boolean_list | (obj_median_slope > slope_max)
    morph_mask_raw = _remove_objects_by_flags(boolean_list, objSum, labeled, CM)
    return np.where(morph_mask_raw > 0, 1, 0).astype(np.uint8)


def low_objects_intersecting_high(high_mask, low_mask):
    """
    Keep only low-probability objects that overlap at least one high-probability object.
    This is the default way flood-water extent is taken from the nested masks.
    """
    high_mask = np.asarray(high_mask, dtype=np.uint8)
    low_mask = np.asarray(low_mask, dtype=np.uint8)
    low_labeled, num_low = ndimage.label(low_mask, structure=ndimage.generate_binary_structure(2, 2))
    if num_low == 0:
        return low_mask
    labels_1based = np.arange(1, num_low + 1, dtype=np.int64)
    # Sum of high_mask over each low object: overlap count per low label
    overlap_count = ndimage.sum(high_mask.astype(np.float64), low_labeled, index=labels_1based)
    keep_labels = labels_1based[overlap_count > 0]
    result = np.where(np.isin(low_labeled, keep_labels) & (low_labeled > 0), 1, 0).astype(np.uint8)
    return result


def low_objects_high_ratio_keep(low_mask, high_mask, min_ratio=0.30):
    """
    Keep a low-probability object only if high-probability overlap / object size >= min_ratio.
    Used for herbaceous wetland (ESA 90): CLI --object_pwhw_high_min_ratio (default 0.30).
    """
    low_mask = np.asarray(low_mask, dtype=np.uint8)
    high_mask = np.asarray(high_mask, dtype=np.uint8)
    low_labeled, num_low = ndimage.label(low_mask, structure=ndimage.generate_binary_structure(2, 2))
    if num_low == 0:
        return low_mask
    labels_1based = np.arange(1, num_low + 1, dtype=np.int64)
    obj_sizes = ndimage.sum(np.ones_like(low_labeled, dtype=np.float64), low_labeled, index=labels_1based)
    overlap_count = ndimage.sum(high_mask.astype(np.float64), low_labeled, index=labels_1based)
    ratio = np.where(obj_sizes > 0, overlap_count / obj_sizes, 0.0)
    keep_labels = labels_1based[ratio >= min_ratio]
    result = np.where(np.isin(low_labeled, keep_labels) & (low_labeled > 0), 1, 0).astype(np.uint8)
    return result


# Fit two Gaussians to an object's SAR intensity histogram (bimodality test).

def _gauss(x, mu, sigma, amplitude):
    return amplitude * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def _bimodal(x, mu1, sigma1, a1, mu2, sigma2, a2):
    return _gauss(x, mu1, sigma1, a1) + _gauss(x, mu2, sigma2, a2)


def _compute_bimodality_metrics(int_db, hist_min=-32, hist_max=-5, num_bins=100):
    """
    Fit two Gaussians to the dB intensity histogram. Return (ashman, surface_ratio, bm_coeff)
    or (None, None, None) if the fit fails. Used to test whether a candidate body still
    looks like a water / land mix in SAR.
    """
    int_db = np.asarray(int_db, dtype=np.float64).ravel()
    int_db = int_db[np.isfinite(int_db)]
    if len(int_db) < 4:
        return None, None, None
    bins = np.linspace(hist_min, hist_max, num_bins + 1)
    counts, _ = np.histogram(int_db, bins=bins, density=True)
    bincenter = (bins[:-1] + bins[1:]) / 2
    binstep = bins[1] - bins[0]
    prob = counts * binstep
    # Initial split at median
    th = np.median(int_db)
    left = int_db[int_db < th]
    right = int_db[int_db > th]
    if len(left) < 2 or len(right) < 2:
        return None, None, None
    mean_lt, std_lt = np.mean(left), max(np.std(left), 1e-10)
    mean_gt, std_gt = np.mean(right), max(np.std(right), 1e-10)
    amp_lt = prob[np.argmin(np.abs(bincenter - mean_lt))] if np.any(np.isfinite(prob)) else 0.5
    amp_gt = prob[np.argmin(np.abs(bincenter - mean_gt))] if np.any(np.isfinite(prob)) else 0.5
    try:
        (mu1, s1, a1, mu2, s2, a2), _ = curve_fit(
            _bimodal, bincenter, prob,
            p0=(mean_lt, std_lt, amp_lt, mean_gt, std_gt, amp_gt),
            bounds=(
                (hist_min, 1e-10, 0, hist_min, 1e-10, 0),
                (hist_max, 10, 1, hist_max, 10, 1),
            ),
            max_nfev=500,
        )
        if mu1 > mu2:
            mu1, s1, a1, mu2, s2, a2 = mu2, s2, a2, mu1, s1, a1
        first_mode, second_mode = (mu1, s1, a1), (mu2, s2, a2)
        ashman = np.sqrt(2) * np.abs(first_mode[0] - second_mode[0]) / np.sqrt(first_mode[1] ** 2 + second_mode[1] ** 2)
        simul_first = _gauss(bincenter, *first_mode)
        simul_second = _gauss(bincenter, *second_mode)
        area_first = np.sum(simul_first)
        area_second = np.sum(simul_second)
        surface_ratio = np.nanmin([area_first, area_second]) / np.max([area_first, area_second]) if (area_first > 0 and area_second > 0) else 0
        # Simple bimodality coefficient (Sarle's, or valley depth)
        simul_all = simul_first + simul_second
        valley_idx = np.argmin(simul_all)
        if simul_all[valley_idx] > 0 and np.max(simul_all) > 0:
            bm = 1 - simul_all[valley_idx] / np.max(simul_all)
        else:
            bm = 0.5
        return float(ashman), float(surface_ratio), float(bm)
    except (RuntimeError, ValueError):
        return None, None, None


def _process_one_bimodality(
    lab,
    labeled,
    intensity_db,
    ref_land,
    ref_land_portion_thresh,
    min_pixel,
    ashman_thresh,
    surface_ratio_thresh,
    bm_thresh,
    hist_min,
    hist_max,
):
    """One labeled object: return (lab, True) if it should be dropped by the bimodality test."""
    watermask = (labeled == lab)
    sizes = int(np.sum(watermask))
    ref_land_portion = float(np.mean(ref_land[watermask])) if watermask.any() else 0.0
    if sizes < min_pixel:
        return (lab, ref_land_portion >= 1.0)
    if ref_land_portion <= ref_land_portion_thresh:
        return (lab, False)
    margin = max(1, int((np.sqrt(2) - 1.2) * np.sqrt(sizes)))
    try:
        mask_buffer = ndimage.binary_dilation(watermask, iterations=margin)
    except Exception:
        mask_buffer = watermask
    intensity_center = np.nanmedian(intensity_db[watermask])
    boundary_mask = mask_buffer & (~watermask)
    intensity_adjacent = intensity_db[boundary_mask]
    intensity_adjacent = intensity_adjacent[np.isfinite(intensity_adjacent)]
    if intensity_adjacent.size == 0:
        return (lab, True)
    if intensity_center > np.nanpercentile(intensity_adjacent, 15):
        return (lab, False)
    intensity_array = intensity_db[mask_buffer]
    intensity_array = intensity_array[np.isfinite(intensity_array)]
    if len(intensity_array) < 4:
        return (lab, True)
    ashman, surface_ratio, bm = _compute_bimodality_metrics(
        intensity_array, hist_min=hist_min, hist_max=hist_max
    )
    if ashman is None:
        return (lab, True)
    bimodal = (ashman > ashman_thresh and surface_ratio > surface_ratio_thresh and bm > bm_thresh)
    return (lab, not bimodal)


def refine_bimodality(
    mask,
    img_lp,
    land_cover,
    ref_land_portion_thresh=0.8,
    min_pixel=40,
    ashman_thresh=1.5,
    surface_ratio_thresh=0.1,
    bm_thresh=0.7,
    hist_min=-32,
    hist_max=-5,
):
    """
    For objects that sit mostly on land-cover (not water class 80), test whether the
    SAR intensity histogram is bimodal (water vs land). Drop the object if it is not.
    Applied only to the "checking" flood-water set, not to preserved large water bodies.
    """
    LCC = np.asarray(land_cover, dtype=np.float64)
    ref_land = ((LCC != 80) & (LCC != 0) & np.isfinite(LCC)).astype(np.uint8)
    eps = 1e-12
    intensity_db = np.where(img_lp > 0, 10.0 * np.log10(np.maximum(img_lp, eps)), np.nan)
    labeled, num_labels = ndimage.label(mask.astype(np.uint8), structure=ndimage.generate_binary_structure(2, 2))
    if num_labels == 0:
        return mask.astype(np.uint8)
    # Vectorized per-label stats (same as Morph-style: one pass)
    labels_1based = np.arange(1, num_labels + 1, dtype=np.int64)
    sizes = ndimage.sum(np.ones_like(labeled, dtype=np.float64), labeled, index=labels_1based)
    ref_land_sums = ndimage.sum(ref_land.astype(np.float64), labeled, index=labels_1based)
    ref_land_portion_arr = np.where(sizes > 0, ref_land_sums / sizes, 0)
    # Small objects: remove if 100% land (no loop)
    remove_set = set()
    for lab in range(1, num_labels + 1):
        if sizes[lab - 1] < min_pixel and ref_land_portion_arr[lab - 1] >= 1.0:
            remove_set.add(lab)
    # Only loop over labels that need bimodality (mostly land, size >= min_pixel)
    to_process = [
        lab for lab in range(1, num_labels + 1)
        if sizes[lab - 1] >= min_pixel and ref_land_portion_arr[lab - 1] > ref_land_portion_thresh
    ]
    for lab in to_process:
        _, remove = _process_one_bimodality(
            lab, labeled, intensity_db, ref_land,
            ref_land_portion_thresh, min_pixel,
            ashman_thresh, surface_ratio_thresh, bm_thresh, hist_min, hist_max,
        )
        if remove:
            remove_set.add(lab)
    keep = (labeled > 0) & (~np.isin(labeled, list(remove_set)))
    return np.where(keep, 1, 0).astype(np.uint8)


# --- Main entry: one object-level step ---

# Fallback HAND / slope cuts if CLI flags are missing (SIENA.py supplies its own defaults).
DEFAULT_HAND_MAX_M = 15.0
DEFAULT_SLOPE_MAX_DEG = 10.0

# Preserve / remove / moderate-clean ratios if CLI flags are missing (see SIENA.py).
DEFAULT_HIGH_MASK_PRESERVE_RATIO = 0.35
# Drop if high_rate < this OR mod_rate < object_remove_mod_max
DEFAULT_REMOVE_HIGH_MAX = 0.075
DEFAULT_REMOVE_MOD_MAX = 0.25
# After bimodality: if high < this AND mod < moderate_clean_mod_max → keep moderate extent only
DEFAULT_MODERATE_CLEAN_HIGH_MAX = 0.15
DEFAULT_MODERATE_CLEAN_MOD_MAX = 0.40


@tim
def run_object_level_process(args, ds):
    """
    Object-level include/exclude for one scene.

    Uses High_mask, Mod_mask, and Low_mask from pixel-level detection.

    Broad steps:
    (1) Read the three probability masks.
    (2) Build permanent-water area (PWA) from ESA land cover + water occurrence;
        split high/low into flood-water (FW) vs PWA.
    (3) On FW only: urban / HAND / slope rules on low-mask objects.
    (4) Desert/snow rules on high, moderate, and low FW masks (CLI desert ratios).
    (5) Default: keep low-FW objects that intersect high-FW objects.
        In PWA: permanent water uses high–low intersection; herbaceous wetland
        requires a minimum high-probability fraction (--object_pwhw_high_min_ratio).
    (6) Classify remaining FW objects as preserve / remove / checking using
        high- and moderate-probability rates inside each object.
    (7) Bimodality test on the checking set; optional moderate-extent clean.
    (8) Final water = FW | PWA result; apply GFM exclusion if present; write SIENA_raw.

    Tunable ratios are CLI flags on SIENA.py (see parse() and README.md).
    """
    maskOut = ds.img_lp <= 0
    valid = np.isfinite(ds.img_lp) & (ds.img_lp > 0)
    nWBClasses = list(getattr(args, "nWBClasses", [50]))
    LCC = np.asarray(ds.land_cover)
    valid_growth = valid & (~np.isin(LCC, nWBClasses))
    rNWB_WST_Org = getattr(args, "rNWB_WST_Org", 0.05)
    hand_max = getattr(args, "object_hand_max_m", DEFAULT_HAND_MAX_M)
    slope_max = getattr(args, "object_slope_max_deg", DEFAULT_SLOPE_MAX_DEG)
    use_region_growth = getattr(args, "use_region_growth", False)

    meta = args.refmeta.copy()
    meta.update(dtype=rasterio.uint8, nodata=255)
    I_pol1 = getattr(ds, "I_pol1", ds.img_lp)
    I_pol1 = np.asarray(I_pol1) if I_pol1 is not None else np.asarray(ds.img_lp)

    # --- 1) High mask (from Pixel_level_process) ---
    if getattr(ds, "High_mask", None) is not None:
        high_mask = np.where(valid, np.asarray(ds.High_mask, dtype=np.uint8), 0).astype(np.uint8)
        high_mask = np.where(high_mask > 0, 1, 0).astype(np.uint8)
    else:
        fuzzy_norm = np.asarray(getattr(ds, "normalized_water_scores", None), dtype=np.float64)
        if fuzzy_norm is None:
            fuzzy_norm = np.zeros_like(ds.img_lp)
        fuzzy_norm = np.where(valid, fuzzy_norm, 0)
        high_mask = np.where(valid & (fuzzy_norm > 0.81), 1, 0).astype(np.uint8)
    n_high = int(np.sum(high_mask))
    print("Object level: high mask: %d pixels (from Pixel_level_process)." % n_high)

    # --- 2) Low mask (from Pixel_level_process) ---
    if getattr(ds, "Low_mask", None) is not None:
        low_mask_raw = np.where(valid, np.asarray(ds.Low_mask, dtype=np.uint8), 0).astype(np.uint8)
        low_mask_raw = np.where(low_mask_raw > 0, 1, 0).astype(np.uint8)
    else:
        fuzzy_norm = np.asarray(getattr(ds, "normalized_water_scores", None), dtype=np.float64)
        if fuzzy_norm is None:
            fuzzy_norm = np.zeros_like(ds.img_lp)
        fuzzy_norm = np.where(valid, fuzzy_norm, 0)
        low_mask_raw = np.where(valid & (fuzzy_norm > 0.51), 1, 0).astype(np.uint8)
    n_low_raw = int(np.sum(low_mask_raw))
    print("Object level: low mask: %d pixels (from Pixel_level_process)." % n_low_raw)

    # Mod mask (from Pixel_level_process) for FW/PW and moderate-clean rules
    if getattr(ds, "Mod_mask", None) is not None:
        Mod_mask = np.where(valid, np.asarray(ds.Mod_mask, dtype=np.uint8), 0).astype(np.uint8)
        Mod_mask = np.where(Mod_mask > 0, 1, 0).astype(np.uint8)
    else:
        Mod_mask = np.zeros_like(high_mask, dtype=np.uint8)

    # Land cover debug (before object rules); intermediate output only when --output_intermediate
    output_intermediate = getattr(args, "output_intermediate", False)
    landcover_arr = np.asarray(ds.land_cover, dtype=np.float64)
    landcover_byte = np.where(valid, np.clip(landcover_arr, 0, 100).astype(np.uint8), 255)
    if output_intermediate:
        with rasterio.open(os.path.join(args.dirOut, "landcover_before_object_rules.tif"), "w", **meta, compress="deflate") as dst:
            dst.write(landcover_byte, 1)

    # --- 2) Build PWA; split high into high_mask_FW, high_mask_PW; write debug masks (PWA definition only here) ---
    wo_th = getattr(args, "permanent_water_wo_threshold", 75.0)
    pwa_lcc = getattr(args, "pwa_lcc", [80, 90])
    if isinstance(pwa_lcc, (int, float)):
        pwa_lcc = [int(pwa_lcc)]
    pwa_lcc = list(pwa_lcc)
    if not pwa_lcc:
        pwa_lcc = [80, 90]
    water_occurrence = getattr(ds, "water_occurrence", None)
    if water_occurrence is not None:
        wo = np.asarray(water_occurrence, dtype=np.float64)
        PWA = (np.isin(LCC, pwa_lcc) | (wo >= wo_th)) & np.isfinite(wo)
    else:
        wo = np.full_like(LCC, np.nan, dtype=np.float64)
        PWA = np.isin(LCC, pwa_lcc)
    PWA = np.where(valid, PWA.astype(np.uint8), 0).astype(np.uint8)
    pwa_bool = PWA.astype(bool)

    high_mask_PW = np.where(high_mask & PWA, 1, 0).astype(np.uint8)
    high_mask_FW = np.where(high_mask & (~pwa_bool), 1, 0).astype(np.uint8)
    low_mask_FW = np.where(low_mask_raw & (~pwa_bool), 1, 0).astype(np.uint8)
    low_cleaned_PW = np.where(low_mask_raw & pwa_bool, 1, 0).astype(np.uint8)
    if output_intermediate:
        for name, arr in [("high_mask_FW", high_mask_FW), ("high_mask_PW", high_mask_PW), ("PWA", PWA)]:
            p = os.path.join(args.dirOut, "%s.tif" % name)
            arr_write = np.where(I_pol1 > 0, arr.astype(np.uint8), 255)
            with rasterio.open(p, "w", **meta, compress="deflate") as dst:
                dst.write(arr_write.astype(meta.get("dtype", rasterio.uint8)), 1)
        print("PWA built; wrote high_mask_FW.tif, high_mask_PW.tif, PWA.tif")

    # --- 3) Object rules on low_mask_FW only (no rules inside PWA) -> low_cleaned_FW ---
    hand_arr = np.asarray(ds.HAND_d_m["arr"], dtype=np.float64) if getattr(ds, "HAND_d_m", None) is not None else None
    dem_arr = np.asarray(ds.DEM_d_m["arr"], dtype=np.float64) if getattr(ds, "DEM_d_m", None) is not None else None
    slope_arr = compute_slope_dem(dem_arr) if dem_arr is not None else None
    print("Object rules on low_mask_FW only (urban, HAND, slope)...")
    t0 = time.time()
    low_cleaned_FW = apply_object_rules(
        low_mask_FW,
        ds.land_cover,
        nWBClasses,
        rNWB_WST_Org=rNWB_WST_Org,
        min_pixels=3,
        hand=hand_arr,
        hand_max=hand_max,
        slope=slope_arr,
        slope_max=slope_max,
    )
    _, n_low_objs = ndimage.label(low_cleaned_FW, structure=ndimage.generate_binary_structure(2, 2))
    print("Low FW after object rules: %d objects (%.2f s)." % (n_low_objs, time.time() - t0))
    if output_intermediate:
        low_combined_write = np.where(I_pol1 > 0, np.where(low_cleaned_FW | low_cleaned_PW, 1, 0).astype(np.uint8), 255)
        with rasterio.open(os.path.join(args.dirOut, "low_mask_after_object_rules.tif"), "w", **meta, compress="deflate") as dst:
            dst.write(low_combined_write.astype(meta.get("dtype", rasterio.uint8)), 1)

    # --- 4) Desert removal on FW only (high_FW, Mod_FW, low_cleaned_FW); PWA unchanged ---
    Mod_mask_FW = np.where(Mod_mask & (~pwa_bool), 1, 0).astype(np.uint8)
    lcc_bare = getattr(args, "lcc_bare", 60)
    desert_high_ratio = getattr(args, "object_desert_high_ratio", getattr(args, "object_desert_bare_ratio", 0.6))
    desert_mod_ratio = getattr(args, "object_desert_mod_ratio", 0.35)
    desert_low_ratio = getattr(args, "object_desert_low_ratio", 0.45)
    n_high_fw_before = int(np.sum(high_mask_FW))
    high_mask_FW = apply_high_mask_desert_rule(
        high_mask_FW, ds.land_cover,
        bare_class=lcc_bare, snow_ice_class=70,
        bare_ratio_thresh=desert_high_ratio,
    )
    print("Desert/snow removal (FW only, >= %.0f%%): high_FW %d -> %d pixels." % (desert_high_ratio * 100, n_high_fw_before, int(np.sum(high_mask_FW))))
    n_mod_fw_before = int(np.sum(Mod_mask_FW))
    Mod_mask_FW = apply_high_mask_desert_rule(
        Mod_mask_FW, ds.land_cover,
        bare_class=lcc_bare, snow_ice_class=70,
        bare_ratio_thresh=desert_mod_ratio,
    )
    print("Desert/snow removal (FW only, >= %.0f%%): Mod_FW %d -> %d pixels." % (desert_mod_ratio * 100, n_mod_fw_before, int(np.sum(Mod_mask_FW))))
    n_low_fw_before = int(np.sum(low_cleaned_FW))
    low_cleaned_FW = apply_high_mask_desert_rule(
        low_cleaned_FW, ds.land_cover,
        bare_class=lcc_bare, snow_ice_class=70,
        bare_ratio_thresh=desert_low_ratio,
    )
    print("Desert/snow removal (FW only, >= %.0f%%): low_FW %d -> %d pixels." % (desert_low_ratio * 100, n_low_fw_before, int(np.sum(low_cleaned_FW))))

    # --- 5) result_FW; 6) result_PW; 7) before_bimodality = result_FW | result_PW ---
    if use_region_growth:
        # --- Region growth path (kept for later) ---
        fuzzy_norm = np.asarray(getattr(ds, "normalized_water_scores", None), dtype=np.float64)
        if fuzzy_norm is None or not np.any(np.isfinite(fuzzy_norm)):
            fuzzy_norm = np.zeros_like(ds.img_lp)
        fuzzy_norm = np.where(valid, fuzzy_norm, 0)
        high_cleaned = apply_object_rules(high_mask, ds.land_cover, nWBClasses, rNWB_WST_Org=rNWB_WST_Org, min_pixels=3)
        high_cleaned = np.where(np.isin(LCC, nWBClasses), 0, high_cleaned).astype(np.uint8)
        print("Region growing from high seed into fuzzy > 0.51...")
        t1 = time.time()
        grown = region_grow_from_fuzzy(
            fuzzy_norm, high_threshold=0.81, relaxed_threshold=0.51,
            valid_mask=valid_growth, initial_binary=high_cleaned,
        )
        n_grown = int(np.sum(grown))
        print("Region growing done: %d water pixels (%.2f s)." % (n_grown, time.time() - t1))
        before_bimodality = grown
    else:
        # --- Flood water (FW): low_FW objects that intersect high_mask_FW (FW only) ---
        print("Object intersection FW: keep low_FW objects that intersect high_mask_FW...")
        t1 = time.time()
        result_FW = low_objects_intersecting_high(high_mask_FW, low_cleaned_FW)
        n_fw = int(np.sum(result_FW))
        print("FW intersection: %d pixels (%.2f s)." % (n_fw, time.time() - t1))
        # --- Permanent water (PW): split into PWPW (LCC 80 + WO >= wo_th) and PWHW (LCC 90 herbaceous wetland) ---
        # PWPW: simple rule = low objects that intersect high in PWPW
        # PWHW: keep low object only if rate of high within object >= pwhw_high_min_ratio (default 30%%)
        pwhw_high_min_ratio = getattr(args, "object_pwhw_high_min_ratio", 0.30)
        PWPW = pwa_bool & (LCC == 80) & np.isfinite(wo) & (wo >= wo_th)
        PWHW = pwa_bool & (LCC == 90)
        high_mask_PWPW = np.where(high_mask_PW & PWPW, 1, 0).astype(np.uint8)
        low_mask_PWPW = np.where(low_cleaned_PW & PWPW, 1, 0).astype(np.uint8)
        high_mask_PWHW = np.where(high_mask_PW & PWHW, 1, 0).astype(np.uint8)
        low_mask_PWHW = np.where(low_cleaned_PW & PWHW, 1, 0).astype(np.uint8)
        t2 = time.time()
        result_PWPW = low_objects_intersecting_high(high_mask_PWPW, low_mask_PWPW)
        result_PWHW = low_objects_high_ratio_keep(low_mask_PWHW, high_mask_PWHW, min_ratio=pwhw_high_min_ratio)
        result_PW = np.where(result_PWPW | result_PWHW, 1, 0).astype(np.uint8)
        n_pw = int(np.sum(result_PW))
        print("PW (PWPW intersect + PWHW high>=%.0f%%): %d pixels (%.2f s)." % (pwhw_high_min_ratio * 100, n_pw, time.time() - t2))
        before_bimodality = np.where(result_FW | result_PW, 1, 0).astype(np.uint8)
        n_intersect = int(np.sum(before_bimodality))
        _, n_objs = ndimage.label(before_bimodality, structure=ndimage.generate_binary_structure(2, 2))
        print("Combined (FW | PW): %d water pixels, %d objects." % (n_intersect, n_objs))

    if output_intermediate:
        before_bimodality_write = np.where(I_pol1 > 0, before_bimodality.astype(np.uint8), 255)
        before_bimodality_path = os.path.join(args.dirOut, "before_bimodality.tif")
        with rasterio.open(before_bimodality_path, "w", **meta, compress="deflate") as dst:
            dst.write(before_bimodality_write.astype(meta.get("dtype", rasterio.uint8)), 1)
        print("Wrote intermediate mask: %s" % before_bimodality_path)

    # --- 7b) Split before_bimodality into FW (exclude PWA) and PW (PWA only); use desert-cleaned FW masks ---
    before_bimodality_low_FW = np.where(before_bimodality & (~pwa_bool), 1, 0).astype(np.uint8)
    before_bimodality_low_PW = np.where(before_bimodality & pwa_bool, 1, 0).astype(np.uint8)
    before_bimodality_High_FW = high_mask_FW
    before_bimodality_Moderate_FW = Mod_mask_FW
    before_bimodality_High_PW = np.where(high_mask & pwa_bool, 1, 0).astype(np.uint8)
    before_bimodality_Moderate_PW = np.where(Mod_mask & pwa_bool, 1, 0).astype(np.uint8)
    if output_intermediate:
        for name, arr in [
            ("before_bimodality_low_FW", before_bimodality_low_FW),
            ("before_bimodality_low_PW", before_bimodality_low_PW),
            ("before_bimodality_High_FW", before_bimodality_High_FW),
            ("before_bimodality_Moderate_FW", before_bimodality_Moderate_FW),
            ("before_bimodality_High_PW", before_bimodality_High_PW),
            ("before_bimodality_Moderate_PW", before_bimodality_Moderate_PW),
        ]:
            path = os.path.join(args.dirOut, "%s.tif" % name)
            arr_w = np.where(I_pol1 > 0, arr.astype(np.uint8), 255)
            with rasterio.open(path, "w", **meta, compress="deflate") as dst:
                dst.write(arr_w.astype(meta.get("dtype", rasterio.uint8)), 1)
        print("Wrote before_bimodality_*_FW and *_PW masks.")

    # --- 9) Before bimodality: only Preserve / Remove / Checking (no moderate_clean yet) ---
    remove_high_max = getattr(args, "object_remove_high_max", DEFAULT_REMOVE_HIGH_MAX)
    remove_mod_max = getattr(args, "object_remove_mod_max", DEFAULT_REMOVE_MOD_MAX)
    high_mask_preserve_ratio = getattr(args, "object_high_mask_preserve_ratio", DEFAULT_HIGH_MASK_PRESERVE_RATIO)

    labeled_fw, num_fw = ndimage.label(before_bimodality_low_FW.astype(np.uint8), structure=ndimage.generate_binary_structure(2, 2))
    labels_1based = np.arange(1, num_fw + 1, dtype=np.int64)
    obj_sizes = ndimage.sum(np.ones_like(labeled_fw, dtype=np.float64), labeled_fw, index=labels_1based)
    high_in_obj = ndimage.sum(before_bimodality_High_FW.astype(np.float64), labeled_fw, index=labels_1based)
    mod_in_obj = ndimage.sum(before_bimodality_Moderate_FW.astype(np.float64), labeled_fw, index=labels_1based)
    ratio_high = np.where(obj_sizes > 0, high_in_obj / obj_sizes, 0.0)
    ratio_mod = np.where(obj_sizes > 0, mod_in_obj / obj_sizes, 0.0)

    # Remove / preserve using CLI ratios (defaults from SIENA.py parse()).
    remove_labels = set(labels_1based[(ratio_high < remove_high_max) | (ratio_mod < remove_mod_max)])
    # Preserve: high-probability rate >= preserve ratio (and not already removed)
    preserved_labels = set(labels_1based[ratio_high >= high_mask_preserve_ratio]) - remove_labels
    # Checking: rest (all go to bimodality; moderate_clean is decided after bimodality)
    checking_labels = set(labels_1based) - remove_labels - preserved_labels

    preserved_FW = np.where(np.isin(labeled_fw, list(preserved_labels)), 1, 0).astype(np.uint8)
    checking_FW = np.where(np.isin(labeled_fw, list(checking_labels)), 1, 0).astype(np.uint8)
    removed_FW = np.where(np.isin(labeled_fw, list(remove_labels)), 1, 0).astype(np.uint8)

    n_pres = int(np.sum(preserved_FW))
    n_check = int(np.sum(checking_FW))
    n_rem = int(np.sum(removed_FW))
    print("FW classification (before bimodality): Preserve %d obj (%d px), Remove %d obj (%d px), Checking %d obj (%d px)." % (
        len(preserved_labels), n_pres, len(remove_labels), n_rem, len(checking_labels), n_check))
    if output_intermediate:
        for p, arr in [
            ("before_bimodality_preserved_FW", preserved_FW),
            ("before_bimodality_checking_FW", checking_FW),
            ("before_bimodality_removed_FW", removed_FW),
        ]:
            path = os.path.join(args.dirOut, "%s.tif" % p)
            arr_w = np.where(I_pol1 > 0, arr.astype(np.uint8), 255)
            with rasterio.open(path, "w", **meta, compress="deflate") as dst:
                dst.write(arr_w.astype(meta.get("dtype", rasterio.uint8)), 1)

    # --- 10) Bimodality refinement only on checking_FW -> bimodality_checked (temporal Low_mask_FW) ---
    print("Bimodality refinement (checking_FW only)...")
    t2 = time.time()
    bimodality_checked = refine_bimodality(
        checking_FW,
        np.asarray(ds.img_lp, dtype=np.float64),
        ds.land_cover,
        ref_land_portion_thresh=0.8,
        min_pixel=40,
        ashman_thresh=1.5,
        surface_ratio_thresh=0.1,
        bm_thresh=0.7,
    )
    _, n_checked_objs = ndimage.label(bimodality_checked, structure=ndimage.generate_binary_structure(2, 2))
    print("Bimodality done: %d water bodies from checking (%.2f s)." % (n_checked_objs, time.time() - t2))

    # --- 11) After bimodality: label bimodality_checked; for each object, overlap with High_FW and Moderate_FW ---
    # If high < 15% AND mod < 40% (AND): use only object ∩ Moderate_FW (moderate_clean). Else: use full object.
    moderate_clean_high_max = getattr(args, "object_moderate_clean_high_max", DEFAULT_MODERATE_CLEAN_HIGH_MAX)
    moderate_clean_mod_max = getattr(args, "object_moderate_clean_mod_max", DEFAULT_MODERATE_CLEAN_MOD_MAX)

    labeled_bc, num_bc = ndimage.label(bimodality_checked.astype(np.uint8), structure=ndimage.generate_binary_structure(2, 2))
    labels_bc = np.arange(1, num_bc + 1, dtype=np.int64)
    bc_sizes = ndimage.sum(np.ones_like(labeled_bc, dtype=np.float64), labeled_bc, index=labels_bc)
    high_in_bc = ndimage.sum(before_bimodality_High_FW.astype(np.float64), labeled_bc, index=labels_bc)
    mod_in_bc = ndimage.sum(before_bimodality_Moderate_FW.astype(np.float64), labeled_bc, index=labels_bc)
    bc_ratio_high = np.where(bc_sizes > 0, high_in_bc / bc_sizes, 0.0)
    bc_ratio_mod = np.where(bc_sizes > 0, mod_in_bc / bc_sizes, 0.0)

    # Moderate-clean: high < 15% AND mod < 40% -> use object ∩ Moderate_FW only
    moderate_clean_bc_labels = set(labels_bc[(bc_ratio_high < moderate_clean_high_max) & (bc_ratio_mod < moderate_clean_mod_max)])
    # Contribution from bimodality_checked: for moderate_clean objects use Moderate_FW extent; else full object
    moderate_clean_extent = np.where(
        np.isin(labeled_bc, list(moderate_clean_bc_labels)) & before_bimodality_Moderate_FW, 1, 0
    ).astype(np.uint8)
    bimodality_keep_full = np.where(
        np.isin(labeled_bc, list(set(labels_bc) - moderate_clean_bc_labels)), 1, 0
    ).astype(np.uint8)
    final_mask_FW = np.where(preserved_FW | moderate_clean_extent | bimodality_keep_full, 1, 0).astype(np.uint8)
    print("After bimodality: %d objects with high<%.0f%% and mod<%.0f%% (AND) -> moderate extent only; rest keep full." % (
        len(moderate_clean_bc_labels), moderate_clean_high_max * 100, moderate_clean_mod_max * 100))
    if output_intermediate:
        mod_ext_path = os.path.join(args.dirOut, "moderate_clean_extent.tif")
        mod_ext_w = np.where(I_pol1 > 0, moderate_clean_extent.astype(np.uint8), 255)
        with rasterio.open(mod_ext_path, "w", **meta, compress="deflate") as dst:
            dst.write(mod_ext_w.astype(meta.get("dtype", rasterio.uint8)), 1)

    # --- 12) final_mask_PW = before_bimodality_low_PW; Final water mask = final_mask_FW | final_mask_PW ---
    final_mask_PW = before_bimodality_low_PW
    water_mask = np.where(final_mask_FW | final_mask_PW, 1, 0).astype(np.uint8)
    _, n_final = ndimage.label(water_mask, structure=ndimage.generate_binary_structure(2, 2))
    print("Final: FW %d px + PW %d px -> %d water bodies." % (int(np.sum(final_mask_FW)), int(np.sum(final_mask_PW)), n_final))
    water_mask = np.where(maskOut, 0, water_mask).astype(np.uint8)
    water_mask = np.where(water_mask > 0, 1, 0).astype(np.uint8)
    # Same permanent-water mask as SIENA_*_RGB_*.tif (quick_look.flood_per_pre_save / png_look)
    WOP_rgb = (
        np.asarray(ds.water_occurrence, dtype=np.float64)
        if getattr(ds, "water_occurrence", None) is not None
        else np.full_like(LCC, np.nan, dtype=np.float64)
    )
    mask_pw_rgb = np.logical_or(WOP_rgb >= wo_th, LCC == 80)
    water_mask = np.where(valid & (mask_pw_rgb | water_mask.astype(bool)), 1, 0).astype(np.uint8)
    mask_profile = args.refmeta.copy()
    mask_profile.update(dtype=rasterio.uint8, nodata=255)
    I_pol1 = getattr(ds, "I_pol1", ds.img_lp)
    if I_pol1 is not None:
        I_pol1 = np.asarray(I_pol1)
    else:
        I_pol1 = np.asarray(ds.img_lp)
    mask_write = np.where(I_pol1 > 0, water_mask, mask_profile["nodata"]).astype(mask_profile["dtype"])
    granules = getattr(args, "granules", "out")
    siena_raw_name = "SIENA_raw_" + granules + ".tif"
    siena_raw_path = os.path.join(args.dirOut, siena_raw_name)
    ds.Morph_mask = water_mask
    ds.Compensate_mask = water_mask
    # GFM exclusion (skip if not available)
    gfm_exclusion = getattr(ds, "gfm_sar_exclusion", None)
    if gfm_exclusion is not None:
        from utils import write_raster
        excl_arr = np.asarray(gfm_exclusion)
        excluded = np.where((water_mask == 1) & (excl_arr > 0), 0, water_mask).astype(np.uint8)
        excluded = np.where(valid & mask_pw_rgb, 1, excluded).astype(np.uint8)
        if output_intermediate:
            file_excluded = os.path.join(args.dirOut, "Excluded_Compensate.tif")
            with rasterio.open(file_excluded, "w", **mask_profile, compress="deflate") as dst:
                dst.write(
                    np.where(I_pol1 > 0, excluded, mask_profile["nodata"]).astype(mask_profile["dtype"]),
                    indexes=1,
                )
            args.fileOutExcludedCompensate = file_excluded
            args.fileOutCompensate = file_excluded
            ds.Compensate_mask = excluded
            gfm_d_m = getattr(ds, "GFM_exclusion_d_m", None)
            if gfm_d_m is not None:
                args.fileOutGFMExclusion = os.path.join(args.dirOut, "GFM_exclusion_mask.tif")
                write_raster(gfm_d_m, args.fileOutGFMExclusion)
        else:
            with rasterio.open(siena_raw_path, "w", **mask_profile, compress="deflate") as dst:
                dst.write(
                    np.where(I_pol1 > 0, excluded, mask_profile["nodata"]).astype(mask_profile["dtype"]),
                    indexes=1,
                )
            args.fileOutCompensate = siena_raw_path
            ds.Compensate_mask = excluded
    else:
        if output_intermediate:
            out_path = os.path.join(args.dirOut, "Compensate.tif")
            with rasterio.open(out_path, "w", **mask_profile, compress="deflate") as dst:
                dst.write(mask_write, indexes=1)
            args.fileOutCompensate = out_path
        else:
            with rasterio.open(siena_raw_path, "w", **mask_profile, compress="deflate") as dst:
                dst.write(mask_write, indexes=1)
            args.fileOutCompensate = siena_raw_path
    return args, ds
