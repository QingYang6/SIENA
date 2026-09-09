#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download a GFM (Global Flood Monitoring) SAR exclusion mask and reproject it
onto the SIENA SAR grid. Used to drop pixels where SAR water mapping is
unreliable (e.g. radar shadow, snow, urban).

Default source is the public EODC STAC API (no extra credentials). An optional
ECMWF MARS path exists for sites that already have that access; SIENA.py uses
EODC only.

Standalone:
  python gfm_sar_exclusion_mars.py /path/to/SAR.tif [--out exclusion.tif]
"""

from __future__ import print_function

import os
import sys
import tempfile
import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from pystac_client import Client

# cfgrib/xarray are imported lazily in _read_grib_to_array_and_geo (they require
# the ecCodes C library: conda install -c conda-forge eccodes cfgrib xarray)

# Default MARS request for CEMS GFM SAR FIM exclusion mask.
# You must replace with the actual stream/class/param from CEMS/ECMWF documentation.
# MARS area format: "N/W/S/E" in degrees (e.g. "50/0/40/10").
# See: https://confluence.ecmwf.int/display/CEMS/MARS
MARS_REQUEST = {
    "class": "ce",           # CEMS class (confirm from CEMS MARS docs)
    "stream": "gfm",         # or product-specific stream
    "expver": "1",
    "type": "em",            # or type for exclusion mask
    "levtype": "sfc",
    "param": "260015",       # placeholder; get real param ID for SAR exclusion mask
    "date": "-1",            # latest; or YYYYMMDD
    "time": "00",
    "target": "gfm_sar_exclusion.grib",
}
# Override from env: JSON path or key=val (e.g. MARS_REQUEST_PARAM=260015)
MARS_AREA_FROM_BOUNDS = True  # use SAR bounds as area=N/W/S/E


def _wgs84_bounds_from_meta(input_data_meta):
    """Return (west, south, east, north) in WGS84 from input_data_meta (raster dict)."""
    meta = input_data_meta["meta"]
    bounds = input_data_meta["bounds"]
    return transform_bounds(meta["crs"], "EPSG:4326", *bounds)


# Tolerance for bounds comparison (degrees)
_BOUNDS_TOL = 1e-6


def _bounds_contain(outer, inner, tol=_BOUNDS_TOL):
    """True if outer (w,s,e,n) fully contains inner (w,s,e,n)."""
    ow, os_, oe, on_ = outer
    iw, is_, ie, in_ = inner
    return (ow <= iw + tol and os_ <= is_ + tol and oe >= ie - tol and on_ >= in_ - tol)


def _bounds_intersect(a, b, tol=_BOUNDS_TOL):
    """True if bounding boxes a and b overlap."""
    aw, as_, ae, an_ = a
    bw, bs_, be, bn_ = b
    return not (ae < bw - tol or be < aw - tol or an_ < bs_ - tol or bn_ < as_ - tol)


def _union_bounds(bounds_list):
    """Union of (w,s,e,n) list -> (w, s, e, n)."""
    if not bounds_list:
        return None
    ws = [b[0] for b in bounds_list]
    ss = [b[1] for b in bounds_list]
    es = [b[2] for b in bounds_list]
    ns = [b[3] for b in bounds_list]
    return (min(ws), min(ss), max(es), max(ns))


def _find_cached_gfm_tiles_covering_bounds(cache_dir, bounds_wgs84):
    """
    Look in cache_dir for GFM exclusion GeoTIFFs that cover the requested bounds.
    Reuse by location: if we already have tile(s) covering this area, skip download.
    Returns list of local paths (empty if no usable cache); paths can be passed to
    _reproject_and_merge_raster_paths.
    """
    if not cache_dir or not os.path.isdir(cache_dir):
        return []
    import glob
    candidates = []
    for ext in ("*.tif", "*.tiff"):
        for path in glob.glob(os.path.join(cache_dir, ext)):
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                continue
            try:
                with rasterio.open(path) as src:
                    cb = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                candidates.append((path, cb))
            except Exception:
                continue
    if not candidates:
        return []

    # Prefer single file that fully contains request
    for path, cb in candidates:
        if _bounds_contain(cb, bounds_wgs84):
            return [path]

    # Else: set of cached tiles whose union contains request (avoid re-download when coverage exists)
    overlapping = [(p, b) for p, b in candidates if _bounds_intersect(b, bounds_wgs84)]
    if not overlapping:
        return []
    union = _union_bounds([b for _, b in overlapping])
    if _bounds_contain(union, bounds_wgs84):
        return [p for p, _ in overlapping]
    return []


def _mars_area_string(bounds_wgs84):
    """MARS area: N/W/S/E in degrees."""
    west, south, east, north = bounds_wgs84
    return "/".join([f"{north:.2f}", f"{west:.2f}", f"{south:.2f}", f"{east:.2f}"])


# EODC GFM STAC API (no MARS needed)
EODC_STAC_URL = "https://stac.eodc.eu/api/v1"
GFM_COLLECTION = "GFM"
# Asset key for exclusion mask in GFM STAC items; try in order if first missing
EODC_EXCLUSION_ASSET_KEYS = ["exclusion_mask", "exclusion_mask_sar", "reference_exclusion_mask"]


def fetch_gfm_exclusion_from_eodc_stac(bounds_wgs84, datetime_range=None, asset_key=None, out_dir=None):
    """
    Download GFM exclusion mask from EODC STAC API (no ECMWF MARS needed).
    bounds_wgs84: (west, south, east, north) in degrees.
    datetime_range: optional "YYYY-MM-DD/YYYY-MM-DD" or None for latest.
    asset_key: STAC asset key (default: first available from EODC_EXCLUSION_ASSET_KEYS).
    out_dir: directory to save tiles; if None, use temp dir.

    Returns list of local GeoTIFF paths (one per STAC item/tile).
    """
    west, south, east, north = bounds_wgs84
    bbox = [west, south, east, north]
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="gfm_exclusion_")
    os.makedirs(out_dir, exist_ok=True)

    catalog = Client.open(EODC_STAC_URL)
    kwargs = dict(collections=[GFM_COLLECTION], bbox=bbox, max_items=500)
    if datetime_range:
        kwargs["datetime"] = datetime_range
    search = catalog.search(**kwargs)
    items = list(search.items())
    if not items:
        raise RuntimeError(
            "No GFM items found for bbox %s (and datetime=%s). Check https://stac.eodc.eu or try a larger area."
            % (bbox, datetime_range)
        )

    # Resolve asset key: use first that exists in at least one item (collection has "exclusion_mask")
    keys_to_try = [asset_key] if asset_key else EODC_EXCLUSION_ASSET_KEYS
    chosen_key = None
    for k in keys_to_try:
        if not k:
            continue
        if any(k in item.assets for item in items):
            chosen_key = k
            break
    if not chosen_key:
        available = list(items[0].assets.keys()) if items else []
        raise RuntimeError(
            "Exclusion mask asset not found in any GFM STAC item. Tried: %s. "
            "Available in first item: %s. Set --asset to one of these."
            % (keys_to_try, available)
        )

    import urllib.request
    local_paths = []
    for i, item in enumerate(items):
        if chosen_key not in item.assets:
            continue
        asset = item.assets[chosen_key]
        href = asset.href
        fname = os.path.basename(href.split("?")[0]) or "tile_%d.tif" % i
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            try:
                urllib.request.urlretrieve(href, path)
            except Exception as e:
                raise RuntimeError("Failed to download %s: %s" % (href[:80], e))
        local_paths.append(path)
    if not local_paths:
        raise RuntimeError(
            "No GFM items in your bbox have the '%s' asset. Try a different area or --datetime."
            % chosen_key
        )
    return local_paths


def _reproject_and_merge_raster_paths(path_list, input_data_meta):
    """Reproject each raster to SAR grid and merge with np.maximum. Returns raster_data_dict.
    Convention: 0 = no exclusion, non-zero (e.g. 1) = exclude. 255 and NaN are normalized to 0
    so EODC nodata/fill does not exclude the whole scene."""
    dst_meta = input_data_meta["meta"].copy()
    dst_crs = dst_meta["crs"]
    dst_transform = dst_meta["transform"]
    dst_height = dst_meta["height"]
    dst_width = dst_meta["width"]
    out = np.zeros((dst_height, dst_width), dtype=np.float32)
    for fp in path_list:
        try:
            with rasterio.open(fp) as src:
                temp = np.zeros((dst_height, dst_width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=temp,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    src_nodata=getattr(src, "nodata", None),
                    dst_nodata=0.0,
                )
                np.maximum(out, temp, out=out)
        except Exception:
            continue
    # Normalize: 255 and NaN often mean nodata in EODC products -> treat as 0 (no exclusion)
    out = np.asarray(out, dtype=np.float32)
    out[np.isnan(out) | ~np.isfinite(out)] = 0.0
    out[out == 255] = 0.0
    # Optional: binarize so output is 0 or 1 (clear convention for GFM_exclusion_mask.tif)
    out[out > 0] = 1.0
    return create_raster_data_dict(out, dst_meta, input_data_meta["bounds"])


def _read_grib_to_array_and_geo(grib_path):
    """
    Read first 2D field from a GRIB file; return (array, transform, crs).
    Uses cfgrib/xarray; requires ecCodes C library (conda install -c conda-forge eccodes).
    """
    import xarray as xr
    import cfgrib  # noqa: F401 - engine for xarray; requires ecCodes C library

    with xr.open_dataset(grib_path, engine="cfgrib") as ds:
        # First data variable (often one 2D field)
        da = None
        for v in ds.data_vars:
            d = ds[v]
            if d.ndim >= 2:
                da = d
                break
        if da is None:
            raise ValueError("No 2D variable found in GRIB: %s" % grib_path)

        arr = np.asarray(da.values, dtype=np.float32)
        if arr.ndim > 2:
            arr = arr.squeeze()
        if arr.ndim != 2:
            raise ValueError("GRIB variable is not 2D: %s" % da.dims)

        # Build rasterio-style transform from lat/lon (copy coords before leaving with block)
        if "latitude" in da.coords and "longitude" in da.coords:
            lats = np.asarray(da.coords["latitude"].values)
            lons = np.asarray(da.coords["longitude"].values)
        elif "lat" in da.coords and "lon" in da.coords:
            lats = np.asarray(da.coords["lat"].values)
            lons = np.asarray(da.coords["lon"].values)
        else:
            raise ValueError("No lat/lon coords in GRIB: %s" % list(da.coords))

        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())

    from rasterio.transform import from_bounds
    h, w = arr.shape
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    crs = rasterio.crs.CRS.from_epsg(4326)
    return arr, transform, crs


def fetch_gfm_sar_exclusion_from_mars(bounds_wgs84, request_override=None, out_grib_path=None):
    """
    Request GFM SAR FIM exclusion mask from MARS for the given WGS84 bounds.
    bounds_wgs84: (west, south, east, north) in degrees.
    request_override: optional dict to merge into MARS_REQUEST (e.g. param, date).
    out_grib_path: optional path to save the GRIB file; otherwise a temp file is used.

    Returns path to the downloaded GRIB file.
    """
    from ecmwfapi import ECMWFService

    req = dict(MARS_REQUEST)
    if request_override:
        req.update(request_override)
    if MARS_AREA_FROM_BOUNDS:
        req["area"] = _mars_area_string(bounds_wgs84)

    if out_grib_path is None:
        fd, out_grib_path = tempfile.mkstemp(suffix=".grib", prefix="gfm_sar_exclusion_")
        os.close(fd)

    # MARS API expects target path as second argument to execute()
    req.pop("target", None)
    server = ECMWFService("mars")
    try:
        server.execute(req, out_grib_path)
    except Exception as e:
        err = str(e).lower()
        if "token" in err and ("disabled" in err or "expired" in err or "invalid" in err):
            raise RuntimeError(
                "ECMWF API key is disabled or expired. Get a new key at https://api.ecmwf.int/v1/key/ "
                "and update ~/.ecmwfapirc. Original error: %s" % e
            )
        if "no access to services/mars" in err or "has no access to" in err and "mars" in err:
            raise RuntimeError(
                "Your ECMWF account does not have MARS access. MARS is restricted; you must request "
                "access (e.g. via https://support.ecmwf.int/ or your institution's ECMWF Computing "
                "Representative). See https://confluence.ecmwf.int/display/CEMS/MARS. Original error: %s" % e
            )
        raise
    return out_grib_path


def create_raster_data_dict(arr, metadata, bounds):
    """Same as utils.create_raster_data_dict for standalone use."""
    return {"arr": arr, "meta": metadata, "bounds": bounds}


def grib_to_raster_data_dict(grib_path, input_data_meta):
    """
    Read GRIB and reproject to the grid defined by input_data_meta.
    Returns same structure as ancillarydata_merge.get_wo(): {'arr', 'meta', 'bounds'}.
    """

    arr_src, src_transform, src_crs = _read_grib_to_array_and_geo(grib_path)
    dst_meta = input_data_meta["meta"].copy()
    dst_crs = dst_meta["crs"]
    dst_transform = dst_meta["transform"]
    dst_height = dst_meta["height"]
    dst_width = dst_meta["width"]

    out = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=arr_src,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=0.0,
    )
    return create_raster_data_dict(out, dst_meta, input_data_meta["bounds"])


def get_gfm_sar_exclusion(
    input_data_meta,
    source="eodc",
    request_override=None,
    cache_dir=None,
    datetime_range=None,
    asset_key=None,
):
    """
    Fetch GFM SAR FIM exclusion mask and return it aligned to the SAR grid.
    Same API as get_wo() / get_lcc() in ancillarydata_merge.py.

    source: "eodc" (default, no MARS) or "mars".
    input_data_meta: dict with 'meta' (rasterio meta), 'bounds' (from ref raster).
    request_override: optional dict for MARS request (ignored if source=eodc).
    cache_dir: if set, save tiles here and reuse by location: before downloading,
        we look for any cached .tif whose bounds cover the request; if found we
        reuse it (one file per area, no redundant download).
    datetime_range: for EODC only, e.g. "2022-09-15/2022-09-16" or None for latest.
    asset_key: for EODC only, STAC asset key for exclusion mask.

    Returns:
        dict with 'arr', 'meta', 'bounds' (float32 raster, 0 = no exclusion, non-zero = exclude).
    """
    bounds_wgs84 = _wgs84_bounds_from_meta(input_data_meta)

    if source == "eodc":
        # Reuse by location: if cache already has tile(s) covering this area, skip download
        if cache_dir:
            cached_paths = _find_cached_gfm_tiles_covering_bounds(cache_dir, bounds_wgs84)
            if cached_paths:
                print("GFM SAR exclusion: reusing %d cached tile(s) covering this area (no download)."
                      % len(cached_paths))
                return _reproject_and_merge_raster_paths(cached_paths, input_data_meta)
        paths = fetch_gfm_exclusion_from_eodc_stac(
            bounds_wgs84,
            datetime_range=datetime_range,
            asset_key=asset_key,
            out_dir=cache_dir,
        )
        return _reproject_and_merge_raster_paths(paths, input_data_meta)

    # MARS
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _mars_area_string(bounds_wgs84).replace(".", "p").replace("/", "_")
        grib_path = os.path.join(cache_dir, "gfm_sar_exclusion_%s.grib" % key)
        if not os.path.isfile(grib_path):
            fetch_gfm_sar_exclusion_from_mars(bounds_wgs84, request_override, grib_path)
    else:
        grib_path = fetch_gfm_sar_exclusion_from_mars(bounds_wgs84, request_override, None)
    try:
        result = grib_to_raster_data_dict(grib_path, input_data_meta)
        return result
    finally:
        if not cache_dir and os.path.isfile(grib_path):
            try:
                os.remove(grib_path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Download GFM SAR exclusion mask (EODC STAC or MARS) and reproject to SAR grid."
    )
    parser.add_argument(
        "sar_reference",
        help="Path to reference SAR GeoTIFF (used for extent and grid)",
    )
    parser.add_argument(
        "out_tif",
        nargs="?",
        default=None,
        help="Optional output GeoTIFF path (same as --out).",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output GeoTIFF path (exclusion mask on SAR grid). If omitted, only download/test.",
    )
    parser.add_argument(
        "--source",
        choices=("eodc", "mars"),
        default="eodc",
        help="Download from EODC STAC (default, no MARS) or MARS.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded tiles or GRIB (optional)",
    )
    parser.add_argument(
        "--asset",
        default=None,
        help="EODC STAC asset key for exclusion mask (e.g. exclusion_mask). Auto-detected if not set.",
    )
    parser.add_argument(
        "--datetime",
        default=None,
        metavar="START/END",
        help="EODC only: time range YYYY-MM-DD/YYYY-MM-DD (default: all/latest).",
    )
    parser.add_argument(
        "--mars-param",
        default=None,
        help="MARS only: override param (e.g. 260015)",
    )
    parser.add_argument(
        "--grib",
        default=None,
        help="Use this GRIB file instead of download (reproject only)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.sar_reference):
        print("Error: SAR reference file not found:", args.sar_reference, file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from utils import read_raster, write_raster

    input_data_meta = read_raster(args.sar_reference)
    request_override = {}
    if args.mars_param:
        request_override["param"] = args.mars_param

    if args.grib:
        if not os.path.isfile(args.grib):
            print("Error: GRIB file not found:", args.grib, file=sys.stderr)
            sys.exit(1)
        print("Using pre-downloaded GRIB (skip MARS):", args.grib)
        try:
            result = grib_to_raster_data_dict(args.grib, input_data_meta)
        except Exception as e:
            print("Error:", e, file=sys.stderr)
            sys.exit(1)
    else:
        print("Fetching GFM SAR exclusion from %s for bounds (WGS84):" % args.source.upper(),
              _wgs84_bounds_from_meta(input_data_meta))
        try:
            result = get_gfm_sar_exclusion(
                input_data_meta,
                source=args.source,
                request_override=request_override or None,
                cache_dir=args.cache_dir,
                datetime_range=args.datetime,
                asset_key=args.asset,
            )
        except Exception as e:
            print("Error:", e, file=sys.stderr)
            sys.exit(1)

    print("Reprojected exclusion mask shape:", result["arr"].shape)

    out_path = args.out or args.out_tif
    if out_path:
        write_raster(result, out_path)
        print("Wrote:", out_path)
    else:
        print("No output path specified; use -o exclusion.tif or pass path as second argument.")


if __name__ == "__main__":
    main()
