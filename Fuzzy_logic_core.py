"""
Per-pixel water likelihood (fuzzy score) from SAR intensity and ancillary layers.

Combines dark backscatter membership with HAND, slope, and water occurrence.
An optional initial water mask from tile Otsu thresholding (initial_threshold_core)
seeds the area term. The score is later cut into high / moderate / low probability
masks in Pixel_level_process.thSeg4.

Fuzzy membership ranges are in get_default_fuzzy_option() (HAND, slope, occurrence).
"""
import numpy as np
import cv2
from utils import create_raster_data_dict, read_raster
from initial_threshold_core import run_initial_threshold

SOBEL_KERNEL_SIZE = 3
PIXEL_RESOLUTION_X = 30
PIXEL_RESOLUTION_Y = 30
RAD_TO_DEG = 180 / np.pi


def compute_slope_dem(dem):
    """Slope in degrees from a DEM using a Sobel kernel (used in fuzzy score and object rules)."""
    sobelx = cv2.Sobel(dem, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    sobely = cv2.Sobel(dem, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
    slope_angle = np.arctan(np.sqrt(
        (sobelx / SOBEL_KERNEL_SIZE / PIXEL_RESOLUTION_X)**2 +
        (sobely / SOBEL_KERNEL_SIZE / PIXEL_RESOLUTION_Y)**2)) * RAD_TO_DEG
    return slope_angle


def zmf(values, minv, maxv):
    """Z-shaped membership: 1 below minv, 0 above maxv (e.g. dark intensity, low HAND)."""
    span = maxv - minv
    if span <= 0 or not np.isfinite(span):
        span = 1.0
        minv = float(minv) if np.isfinite(minv) else 0.0
        maxv = minv + 1.0
    center_value = (minv + maxv) / 2
    output = np.zeros_like(values, dtype='float32')
    membership_left = 1 - 2 * ((values - minv) / span)**2
    membership_right = 2 * ((values - maxv) / span)**2
    mask_left = (values >= minv) & (values <= center_value)
    mask_right = (values > center_value) & (values <= maxv)
    output[mask_left] = membership_left[mask_left]
    output[mask_right] = membership_right[mask_right]
    output[values <= minv] = 1
    output[values >= maxv] = 0
    output[np.isnan(values)] = np.nan
    return output


def smf(values, minv, maxv):
    """S-shaped membership: 0 below minv, 1 above maxv (e.g. large water area, high occurrence)."""
    span = maxv - minv
    if span <= 0 or not np.isfinite(span):
        span = 1.0
        minv = float(minv) if np.isfinite(minv) else 0.0
        maxv = minv + 1.0
    center_value = (minv + maxv) / 2
    output = np.zeros_like(values, dtype='float32')
    membership_left = 2 * ((values - minv) / span)**2
    membership_right = 1 - 2 * ((values - maxv) / span)**2
    mask_left = (values >= minv) & (values <= center_value)
    mask_right = (values > center_value) & (values <= maxv)
    output[mask_left] = membership_left[mask_left]
    output[mask_right] = membership_right[mask_right]
    output[values <= minv] = 0
    output[values >= maxv] = 1
    return output


def calculate_water_area(binary_raster):
    """Connected-component size (pixels) for each water pixel; used as an area membership input."""
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        np.array(binary_raster, dtype=np.uint8), connectivity=8)
    excluded_area_ind = np.unique(output[binary_raster == 0])
    sizes = stats[:, -1]
    sizes = np.delete(sizes, excluded_area_ind)
    nb_components -= 1
    old_val = np.arange(1, nb_components + 1) - 0.1
    kin = np.searchsorted(old_val, output)
    sizes = np.insert(sizes, 0, 0, axis=0)
    size_raster = sizes[kin]
    return size_raster


def normalize_by_range_profile(I, axis=0):
    """
    Normalize intensity by range profile so typical level ~1 along range.
    profile = nanmedian(I, axis=axis); I_corrected = I / profile.
    Simplest: axis=0 -> profile per column (median over rows); use when range is along columns.
    """
    I = np.asarray(I, dtype=np.float64)
    valid = np.isfinite(I) & (I > 0)
    # profile: median along the other dimension(s)
    profile = np.nanmedian(np.where(valid, I, np.nan), axis=axis)
    if profile.ndim == 0:
        profile = np.atleast_1d(profile)
    # avoid div by zero; where profile is 0/nan use 1 so we don't change
    profile = np.where(np.isfinite(profile) & (profile > 0), profile, 1.0)
    if axis == 0:
        # profile shape (cols,) -> broadcast to (rows, cols)
        I_norm = I / np.maximum(profile, 1e-10)
    else:
        # profile shape (rows,) -> broadcast to (rows, cols)
        I_norm = I / np.maximum(profile[:, np.newaxis], 1e-10)
    return np.where(valid, I_norm, I).astype(np.float32)  # leave invalid as original


def get_default_fuzzy_option():
    """HAND, slope, water-occurrence, and area ranges that enter the fuzzy water score."""
    return {
        'dark_area_land': -18,
        'dark_area_water': -21,
        'high_frequent_water_min': 30,
        'high_frequent_water_max': 80,
        'hand_min': 0,
        'hand_max': 15,
        'hand_threshold': 20,
        'slope_min': 0,
        'slope_max': 5,
        'area_min': 3,
        'area_max': 20,
        'reference_water_min': 10,
        'reference_water_max': 90
    }


def get_label_landcover_esa_10():
    """ESA WorldCover class codes used when combining land cover with the fuzzy score."""
    label = dict()
    label['Tree Cover'] = 10
    label['Shrubs'] = 20
    label['Grassland'] = 30
    label['Crop'] = 40
    label['Urban'] = 50
    label['Bare sparse vegetation'] = 60
    label['Snow and Ice'] = 70
    label['Permanent water bodies'] = 80
    label['Herbaceous wetland'] = 90
    label['Mangrove'] = 95
    label['Moss and lichen'] = 100
    label['No_data'] = 0
    return label


def compute_fuzzy_value(
    intensity,
    slope,
    hand,
    landcover,
    landcover_label,
    reference_water,
    pol_list,
    workflow,
    initial_water_mask=None,
):
    """
    Combine SAR dark-area membership with HAND, slope, and water occurrence
    into one fuzzy water score in [0, 1]. workflow selects which ancillary
    terms are averaged (default SIENA path vs an area-weighted variant).
    """
    fuzzy_option = get_default_fuzzy_option()
    pol_list = ['VV', 'VH']
    _, rows, cols = intensity.shape
    intensity_z_set = []
    if initial_water_mask is not None:
        initial_map = np.where(initial_water_mask, 1, 0).astype('uint8')
    else:
        initial_map = np.ones((rows, cols), dtype='uint8')
    low_backscatter_cand = np.ones((rows, cols), dtype=bool)
    dark_water_cand = np.ones((rows, cols), dtype=bool)

    for int_id, pol in enumerate(pol_list):
        # Scene-wide percentiles: "dark" (water-like) is relative to the whole scene.
        peak = np.percentile(intensity[int_id], 25)
        valley = np.percentile(intensity[int_id], 75)
        intensity_band = intensity[int_id, :, :]
        temp = zmf(intensity_band, peak, valley)
        intensity_z_set.append(temp)
        if initial_water_mask is None:
            mask = intensity_band < peak
            initial_map[~mask] = 0
        if pol in ['VH', 'HV']:
            pol_thresh = fuzzy_option['dark_area_land']
            water_thresh = fuzzy_option['dark_area_water']
            low_backscatter = intensity_band < pol_thresh
            low_backscatter_cand &= low_backscatter
            dark_water_cand &= intensity_band < water_thresh

    intensity_z_set = np.array(intensity_z_set)
    landcover_flat = np.isin(landcover, [
        landcover_label['Bare sparse vegetation'],
        landcover_label['Shrubs'],
        landcover_label['Grassland'],
        landcover_label['Herbaceous wetland']])
    flat_area = landcover_flat & (slope < 5) & low_backscatter_cand
    high_frequent_water = (reference_water > fuzzy_option['high_frequent_water_min']) & \
                          (reference_water < fuzzy_option['high_frequent_water_max']) & \
                          low_backscatter_cand
    dark_water = dark_water_cand & (reference_water >= fuzzy_option['high_frequent_water_max'])
    co_pol_ind = next((i for i, p in enumerate(pol_list) if p in ['VV', 'HH']), None)
    cross_pol_ind = next((i for i, p in enumerate(pol_list) if p in ['VH', 'HV']), None)
    if co_pol_ind is not None and cross_pol_ind is not None:
        intensity_z_set[cross_pol_ind][high_frequent_water] = intensity_z_set[co_pol_ind][high_frequent_water]
        intensity_z_set[cross_pol_ind][flat_area] = intensity_z_set[co_pol_ind][flat_area]
        intensity_z_set[co_pol_ind][dark_water] = intensity_z_set[cross_pol_ind][dark_water]
    copol_only = high_frequent_water | flat_area
    nanmean_intensity_z = np.nanmean(intensity_z_set, axis=0)
    hand = np.where(np.isnan(hand), 0, hand)
    hand_z = zmf(hand, fuzzy_option['hand_min'], fuzzy_option['hand_max'])
    slope_z = zmf(slope, fuzzy_option['slope_min'], fuzzy_option['slope_max'])
    handem = hand < fuzzy_option['hand_threshold']
    wbsmask = (initial_map == 1) & handem
    area_map = calculate_water_area(wbsmask)
    area_s = smf(area_map, fuzzy_option['area_min'], fuzzy_option['area_max'])
    ref_water_s = smf(reference_water, fuzzy_option['reference_water_min'], fuzzy_option['reference_water_max'])
    if workflow in ['siena_default', 'siena_ni']:
        ancillary = (hand_z + slope_z + ref_water_s) / 3
    elif workflow == 'twele':
        ancillary = (hand_z + slope_z + area_s) / 3
    else:
        raise ValueError("Unsupported workflow")
    avgvalue = 0.5 * nanmean_intensity_z + 0.5 * ancillary
    return avgvalue, intensity_z_set, hand_z, slope_z, area_s, ref_water_s, copol_only


def run_fuzzy_core(ds):
    """
    Compute fuzzy logic water likelihood. Expects ds with numpy arrays (no Dask).
    When input is not from .nc: normalizes intensity by range profile (median per column)
    and stores ds.img_lp_norm, ds.img_cp_norm. When input is from .nc (ds.from_nc True),
    uses sigma as-is (no range-profile normalization) since the product is already calibrated.
    """
    img_lp = np.asarray(ds.img_lp, dtype=np.float64)
    img_cp = np.asarray(ds.img_cp, dtype=np.float64)
    if getattr(ds, "from_nc", False):
        # .nc input: sigma already linear and calibrated; skip range-profile normalization
        ds.img_lp_norm = np.asarray(ds.img_lp, dtype=np.float32)
        ds.img_cp_norm = np.asarray(ds.img_cp, dtype=np.float32)
    else:
        # GeoTIFF/zip: apply range-profile normalization for flatter radiometry
        ds.img_lp_norm = normalize_by_range_profile(img_lp, axis=0)
        ds.img_cp_norm = normalize_by_range_profile(img_cp, axis=0)
    HAND_d_m = ds.HAND_d_m
    DEM_d_m = ds.DEM_d_m
    dem = np.asarray(DEM_d_m['arr'], dtype=np.float64)
    hand = np.asarray(HAND_d_m['arr'], dtype=np.float64)
    slope = compute_slope_dem(dem)
    valid = img_lp > 0
    dem = np.where(valid, dem, np.nan)
    hand = np.where(valid, hand, np.nan)
    slope = np.where(valid, slope, np.nan)
    # Use normalized intensity for fuzzy (25/75 and zmf)
    intensity = np.stack(
        [
            np.asarray(ds.img_lp_norm, dtype=np.float64),
            np.asarray(ds.img_cp_norm, dtype=np.float64),
        ],
        axis=0,
    )
    pol_list = ['VV', 'VH']
    reference_water = np.where(valid, ds.water_occurrence, np.nan)
    reference_water = np.where(reference_water == 255, 100, reference_water)
    landcover = np.where(valid, ds.land_cover, np.nan)
    landcover_label = get_label_landcover_esa_10()
    workflow = 'siena_default'

    # Initial water mask from tile Otsu thresholds. If that step cannot find
    # mixed land/water tiles, fuzzy scoring continues without it.
    initial_water_mask = None
    print("Running initial thresholding (permanent water ref >= 90, Otsu per tile)...")
    import time
    t0 = time.time()
    try:
        thr_result = run_initial_threshold(intensity=intensity, reference_water=reference_water)
        elapsed = time.time() - t0
        if thr_result is not None and thr_result.get("initial_water_mask") is not None:
            initial_water_mask = thr_result["initial_water_mask"]
            perm_ref = thr_result.get("permanent_water_ref", 90.0)
            print("Initial thresholding success (permanent_water ref >= %.0f, Otsu per tile) in %.2f s." % (perm_ref, elapsed))
        else:
            reason = thr_result.get("reason", "unknown") if isinstance(thr_result, dict) else "no result"
            print("Initial thresholding failed: %s (%.2f s)." % (reason, elapsed))
    except Exception as e:
        print("Initial thresholding failed: %s (%.2f s)." % (e, time.time() - t0))
        initial_water_mask = None

    print("Building fuzzy map...")
    avgvalue, intensity_z_set, hand_z, slope_z, area_s, ref_water_s, copol_only = compute_fuzzy_value(
        intensity=intensity,
        slope=slope,
        hand=hand,
        landcover=landcover,
        landcover_label=landcover_label,
        reference_water=reference_water,
        pol_list=pol_list,
        workflow=workflow,
        initial_water_mask=initial_water_mask,
    )
    return {
        'fuzzy_score': avgvalue,
        'intensity_z_set': intensity_z_set,
        'hand_z': hand_z,
        'slope_z': slope_z,
        'area_s': area_s,
        'ref_water_s': ref_water_s,
        'copol_only_mask': copol_only
    }
