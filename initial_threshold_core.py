"""
Tile-based initial water mask from SAR intensity.

Tiles that contain both reference water (high water occurrence) and land are
kept. In each of those tiles, Otsu's method finds an intensity threshold in dB.
The per-tile thresholds are interpolated to the full scene and pixels darker
than the local threshold become the initial water mask used by fuzzy scoring.

Reference water cut: water_occurrence >= 90 (permanent_water_value 0.9).
"""
from __future__ import annotations
import numpy as np
import cv2

# water = reference_water >= PERMANENT_WATER_VALUE * 100  →  90 on a 0–100 scale
PERMANENT_WATER_VALUE = 0.9  # [0-1]


def _otsu_threshold_dB(intensity_db_flat: np.ndarray, n_bins: int = 128) -> float:
    """Otsu threshold on a 1-D dB intensity sample; returns threshold in dB."""
    arr = np.asarray(intensity_db_flat, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return np.nan
    lo, hi = np.nanpercentile(arr, [1, 99])
    if hi <= lo:
        return np.nan
    bins = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(arr, bins=bins)
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.nan
    bin_centers = (bins[:-1] + bins[1:]) / 2
    sum_total = np.sum(counts * bin_centers)
    sum_b = 0
    w_b = 0
    best_var = 0.0
    best_t = bin_centers[0]
    for i in range(len(counts) - 1):
        w_b += counts[i]
        w_f = total - w_b
        if w_b == 0 or w_f == 0:
            continue
        sum_b += counts[i] * bin_centers[i]
        sum_f = sum_total - sum_b
        mu_b = sum_b / w_b
        mu_f = sum_f / w_f
        var_between = w_b * w_f * (mu_b - mu_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = bin_centers[i]
    return float(best_t)


def _compute_tile_thresholds(
    intensity_band_db: np.ndarray,
    ref_water: np.ndarray,
    tile_size: int = 200,
    permanent_water_ref: float = 90.0,
    min_valid_frac: float = 0.1,
    min_samples: int = 20,
) -> np.ndarray:
    """
    For each tile that has both water (ref >= permanent_water_ref) and land
    (ref < that cut), compute an Otsu threshold on SAR intensity (dB).
    Returns (ny, nx) thresholds in dB (np.nan where the tile was skipped).
    """
    band = np.asarray(intensity_band_db, dtype=np.float32)
    ref = np.asarray(ref_water, dtype=np.float32)
    H, W = band.shape
    ny = (H + tile_size - 1) // tile_size
    nx = (W + tile_size - 1) // tile_size
    tile_thr = np.full((ny, nx), np.nan, dtype=np.float32)

    for iy in range(ny):
        y0 = iy * tile_size
        y1 = min((iy + 1) * tile_size, H)
        for ix in range(nx):
            x0 = ix * tile_size
            x1 = min((ix + 1) * tile_size, W)

            b = band[y0:y1, x0:x1]
            r = ref[y0:y1, x0:x1]
            valid = np.isfinite(b) & np.isfinite(r)
            if valid.sum() < min_valid_frac * b.size:
                continue

            water = valid & (r >= permanent_water_ref)
            land = valid & (r < permanent_water_ref)
            if water.sum() < min_samples or land.sum() < min_samples:
                continue

            thr = _otsu_threshold_dB(b[valid])
            if np.isfinite(thr):
                tile_thr[iy, ix] = np.float32(thr)

    return tile_thr


def _upsample_thresholds(tile_thr: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Bilinearly upsample per-tile thresholds to full resolution.
    NaNs are filled with the median of valid tile thresholds before upsampling.
    """
    if tile_thr.size == 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    thr = np.asarray(tile_thr, dtype=np.float32)
    valid = np.isfinite(thr)
    if not np.any(valid):
        return np.full((H, W), np.nan, dtype=np.float32)

    fill_value = np.median(thr[valid])
    thr_filled = np.where(valid, thr, fill_value).astype(np.float32)

    if thr_filled.shape == (1, 1):
        return np.full((H, W), thr_filled[0, 0], dtype=np.float32)

    # cv2.resize expects (W, H) order for size
    up = cv2.resize(thr_filled, (W, H), interpolation=cv2.INTER_LINEAR)
    return up.astype(np.float32)


def run_initial_threshold(
    intensity: np.ndarray,
    reference_water: np.ndarray,
    tile_size: int = 200,
    permanent_water_ref: float = None,
):
    """
    Build an initial water mask from tile Otsu thresholds on SAR intensity.

    - Tile selection: tiles that contain both water (ref_water >= permanent_water_ref)
      and land (ref_water < permanent_water_ref). Default cut is 90 on a 0–100 scale.
    - Threshold: Otsu on SAR intensity (dB) per selected tile (reference water is
      used only to pick tiles, not to set the dB cut).
    - Output: pixels darker than the interpolated threshold map.
    """
    if permanent_water_ref is None:
        permanent_water_ref = PERMANENT_WATER_VALUE * 100.0  # 90
    intensity = np.asarray(intensity, dtype=np.float64)
    ref_water = np.asarray(reference_water, dtype=np.float64)

    if intensity.ndim == 2:
        intensity = intensity[np.newaxis, ...]

    n_pol, H, W = intensity.shape
    eps = 1e-12
    intensity_db = np.where(intensity > 0, 10.0 * np.log10(np.maximum(intensity, eps)), np.nan)

    threshold_maps = []
    for pol_id in range(n_pol):
        band_db = intensity_db[pol_id]
        tile_thr = _compute_tile_thresholds(
            band_db,
            ref_water,
            tile_size=tile_size,
            permanent_water_ref=permanent_water_ref,
        )
        thr_map = _upsample_thresholds(tile_thr, H, W)
        threshold_maps.append(thr_map)

    threshold_maps = np.stack(threshold_maps, axis=0)

    thr_co = threshold_maps[0]
    if not np.any(np.isfinite(thr_co)):
        return {
            "success": False,
            "reason": "no valid co-pol tile thresholds (insufficient land/water mix in tiles)",
        }

    thr_cross = threshold_maps[1] if n_pol > 1 and np.any(np.isfinite(threshold_maps[1])) else None

    co_band_db = intensity_db[0]
    initial_water_mask = np.isfinite(co_band_db) & np.isfinite(thr_co) & (co_band_db < thr_co)

    return {
        "success": True,
        "threshold_map_co": thr_co.astype(np.float32),
        "threshold_map_cross": None if thr_cross is None else thr_cross.astype(np.float32),
        "initial_water_mask": initial_water_mask,
        "permanent_water_ref": permanent_water_ref,
    }
