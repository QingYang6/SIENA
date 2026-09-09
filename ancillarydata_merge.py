#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ancillary layers for SIENA: water occurrence, land cover, DEM, HAND, GFM exclusion.

By default, intersecting tiles are downloaded from public cloud URLs and cached
under ancillary_download/ (or ANCILLARY_ROOT / --ancillary-download-folder).
If the tiles are already on disk they are reused. Set SIENA_OFFLINE=1 to never
open the network (pre-stage the cache first; see check_env.py and README.md).

Entry points used by Load_Data: get_wo, get_lcc, get_dem, get_HAND30, get_gfm_exclusion.
"""

import os
import sys
import time
import tempfile
import requests
import fcntl
import numpy as np
from osgeo import ogr
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
import rasterio.enums as rio_enums
from pyproj import Transformer

from utils import (
    read_raster, reproject_clip, create_raster_data_dict, convert_bounds,
    mergelist, cropfile, getfile_list, intersecting_files, getinterfolder_files,
    getcropfolder_files, write_raster,
)  # read_raster, reproject_clip, create_raster_data_dict, convert_bounds, mergelist, cropfile

# ---- Optional GDAL/HTTP tuning (safe defaults) ----
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.json,.geojson")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "1048576")

# ---- Path setup ----
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Choose a base root for ancillary downloads; configurable for batch runs
BASE_ANC_ROOT = os.environ.get("ANCILLARY_ROOT", script_dir)

GSW_folder    = os.path.join(BASE_ANC_ROOT, 'ancillary_download/GSW_download')
LCC_folder    = os.path.join(BASE_ANC_ROOT, 'ancillary_download/LCC_download')
DEM_folder    = os.path.join(BASE_ANC_ROOT, 'ancillary_download/DEM_download')
HAND30_folder = os.path.join(BASE_ANC_ROOT, 'ancillary_download/HAND30_download')
GFM_exclusion_folder = os.path.join(BASE_ANC_ROOT, 'ancillary_download/GFM_exclusion_download')
for d in (GSW_folder, LCC_folder, DEM_folder, HAND30_folder, GFM_exclusion_folder):
    os.makedirs(d, exist_ok=True)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


# SIENA_OFFLINE: default is off (online). When set (e.g. on compute nodes with no egress),
# use only local ancillary caches — no HTTP.
# Online mode: index GeoJSONs are read from ancillary_download/*/_download/ if present;
# otherwise they are downloaded once into that folder (same paths as check_env.py).
SIENA_OFFLINE = _env_truthy("SIENA_OFFLINE")

# Canonical HTTP URLs for tile-index GeoJSON (cached on disk in online mode).
_INDEX_URL_DEM = "https://asf-dem-west.s3.amazonaws.com/v2/cop30-2021.geojson"
_INDEX_URL_HAND = "https://glo-30-hand.s3.amazonaws.com/v1/2021/glo-30-hand.geojson"
_INDEX_URL_LCC_GRID = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/esa_worldcover_2020_grid.geojson"


def _ancillary_parent() -> str:
    """Parent of *\_download folders (usually .../ancillary_download)."""
    return os.path.dirname(GSW_folder)


def _offline_geojson_dem_index() -> str:
    return os.path.join(DEM_folder, "cop30-2021.geojson")


def _offline_geojson_hand_index() -> str:
    return os.path.join(HAND30_folder, "glo-30-hand.geojson")


def _offline_geojson_lcc_index() -> str:
    return os.path.join(LCC_folder, "esa_worldcover_2020_grid.geojson")


def set_ancillary_download_folder(ancillary_download_dir):
    """
    Use a custom ancillary_download root (the folder that contains GSW_download, LCC_download,
    DEM_download, HAND30_download, GFM_exclusion_download). Default layout at import is
    <ANCILLARY_ROOT or script_dir>/ancillary_download/...
    """
    global GSW_folder, LCC_folder, DEM_folder, HAND30_folder, GFM_exclusion_folder
    root = os.path.abspath(os.path.expanduser(ancillary_download_dir))
    GSW_folder = os.path.join(root, 'GSW_download')
    LCC_folder = os.path.join(root, 'LCC_download')
    DEM_folder = os.path.join(root, 'DEM_download')
    HAND30_folder = os.path.join(root, 'HAND30_download')
    GFM_exclusion_folder = os.path.join(root, 'GFM_exclusion_download')
    for d in (GSW_folder, LCC_folder, DEM_folder, HAND30_folder, GFM_exclusion_folder):
        os.makedirs(d, exist_ok=True)

# -----------------------
# Robust download helpers
# -----------------------
def _acquire_file_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    f = open(lock_path, "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f

def _release_file_lock(f):
    try:
        fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()

def _validate_tif_quick(path: str) -> bool:
    with rasterio.open(path) as src:
        _ = src.count
        _ = (src.width, src.height)
    return True

def download_file(url, folder, retries=4, timeout=30, chunk=1024*1024):
    """Atomic, validated, per-file-locked HTTP download. Idempotent."""
    # Remove the '/vsicurl/' prefix if it exists
    if isinstance(url, str) and url.startswith('/vsicurl/'):
        url = url[len('/vsicurl/'):]
    os.makedirs(folder, exist_ok=True)
    filename = os.path.basename(url.split("?", 1)[0])
    finalpath = os.path.join(folder, filename)
    # Offline HPC: never open network URLs; require pre-staged GeoTIFF tiles.
    if SIENA_OFFLINE:
        if os.path.isfile(finalpath):
            try:
                _validate_tif_quick(finalpath)
                return finalpath
            except Exception:
                try:
                    os.remove(finalpath)
                except OSError:
                    pass
        raise RuntimeError(
            "SIENA_OFFLINE: missing ancillary GeoTIFF (pre-download on a login node with internet): %s\n"
            "Expected under folder: %s\n"
            "Run once online for this tile, or use check_env.py + SIENA on login to warm the cache."
            % (filename, folder)
        )

    lockpath = finalpath + ".lock"

    # Fast path: already present and valid
    if os.path.exists(finalpath):
        try:
            _validate_tif_quick(finalpath)
            return finalpath
        except Exception:
            try: os.remove(finalpath)
            except: pass

    # Serialize writers for this specific file
    lock_fh = _acquire_file_lock(lockpath)
    try:
        # Re-check after we got the lock (another proc may have finished it)
        if os.path.exists(finalpath):
            try:
                _validate_tif_quick(finalpath)
                return finalpath
            except Exception:
                try: os.remove(finalpath)
                except: pass

        # Atomic temp path
        tmp_fd, tmppath = tempfile.mkstemp(prefix=filename+".", suffix=".part", dir=folder)
        os.close(tmp_fd)

        backoff = 2
        for attempt in range(1, retries+1):
            try:
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    expected = int(r.headers.get("Content-Length", "0"))
                    size = 0
                    with open(tmppath, "wb") as out:
                        for c in r.iter_content(chunk_size=chunk):
                            if c:
                                out.write(c)
                                size += len(c)
                if expected and size != expected:
                    raise IOError(f"Incomplete download: got {size}, expected {expected}")

                # Quick TIFF sanity check
                _validate_tif_quick(tmppath)

                # Atomic publish
                os.replace(tmppath, finalpath)
                return finalpath

            except Exception:
                try: os.remove(tmppath)
                except: pass
                if attempt == retries:
                    raise
                time.sleep(backoff)
                backoff = min(backoff*2, 30)
                tmp_fd, tmppath = tempfile.mkstemp(prefix=filename+".", suffix=".part", dir=folder)
                os.close(tmp_fd)
    finally:
        _release_file_lock(lock_fh)
        try: os.remove(lockpath)
        except: pass

def download_files(file_list, local_folder):
    """Keep signature; now uses the robust atomic downloader."""
    local_file_list = []
    for url in file_list:
        local_file_list.append(download_file(url, local_folder))
    return local_file_list


def _ensure_index_geojson(local_path: str, url: str) -> str:
    """
    Online mode: if local_path exists and is non-empty, return it; else download url to local_path.

    Offline mode is handled by callers (SIENA_OFFLINE); this is only used when not offline.
    """
    _dir = os.path.dirname(os.path.abspath(local_path))
    os.makedirs(_dir, exist_ok=True)
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    lockpath = local_path + ".lock"
    lock_fh = _acquire_file_lock(lockpath)
    try:
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        tmp_fd, tmppath = tempfile.mkstemp(
            prefix=os.path.basename(local_path) + ".", suffix=".part", dir=_dir
        )
        os.close(tmp_fd)
        backoff = 2
        try:
            for attempt in range(1, 5):
                try:
                    with requests.get(url, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with open(tmppath, "wb") as out:
                            for c in r.iter_content(chunk_size=1024 * 1024):
                                if c:
                                    out.write(c)
                    os.replace(tmppath, local_path)
                    return local_path
                except Exception:
                    try:
                        os.remove(tmppath)
                    except OSError:
                        pass
                    if attempt == 4:
                        raise
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    tmp_fd, tmppath = tempfile.mkstemp(
                        prefix=os.path.basename(local_path) + ".", suffix=".part", dir=_dir
                    )
                    os.close(tmp_fd)
        finally:
            if os.path.exists(tmppath):
                try:
                    os.remove(tmppath)
                except OSError:
                    pass
    finally:
        _release_file_lock(lock_fh)
        try:
            os.remove(lockpath)
        except OSError:
            pass
    return local_path

# -----------------------
# Robust merge (VRT+merge)
# -----------------------
def reproject_and_merge(file_list, input_data_meta):
    """
    Reproject each source directly into the target grid and np.maximum accumulate.
    This avoids WarpedVRT and merge(), eliminating boundless-read issues entirely.
    """
    dst_meta      = input_data_meta['meta']
    dst_crs       = dst_meta['crs']
    dst_transform = dst_meta['transform']
    dst_width     = dst_meta['width']
    dst_height    = dst_meta['height']

    # Preallocate output array
    out = np.zeros((dst_height, dst_width), dtype=np.float32)

    for fp in file_list:
        try:
            with rasterio.open(fp) as src:
                # Temporary buffer for this tile
                temp = np.zeros((dst_height, dst_width), dtype=np.float32)

                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=temp,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest,
                    src_nodata=getattr(src, "nodata", None),
                    dst_nodata=0,
                )
                # Max mosaic
                np.maximum(out, temp, out=out)
        except Exception:
            # Skip bad tiles; we already validated on download, but be defensive
            continue

    return create_raster_data_dict(out, dst_meta, input_data_meta['bounds'])

# -----------------------
# Dataset entry points
# -----------------------
def get_wo(input_data_meta):
    """Global Surface Water occurrence, clipped and reprojected to the SAR grid."""
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    wo_list = getWO(wgs84_bounds)
    local_wo_list = download_files(wo_list, GSW_folder)
    return reproject_and_merge(local_wo_list, input_data_meta)

def get_lcc(input_data_meta):
    """ESA WorldCover land cover, clipped and reprojected to the SAR grid."""
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    lcc_list = getLCC(wgs84_bounds)
    local_lcc_list = download_files(lcc_list, LCC_folder)
    return reproject_and_merge(local_lcc_list, input_data_meta)

def get_dem(input_data_meta):
    """Copernicus GLO-30 DEM, clipped and reprojected to the SAR grid."""
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    dem_list = glo30list(wgs84_bounds)
    local_dem_list = download_files(dem_list, DEM_folder)
    return reproject_and_merge(local_dem_list, input_data_meta)

def get_HAND30(input_data_meta):
    """GLO-30 height above nearest drainage (HAND), clipped to the SAR grid."""
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    hand_list = HAND30list(wgs84_bounds)
    local_hand_list = download_files(hand_list, HAND30_folder)
    return reproject_and_merge(local_hand_list, input_data_meta)

def get_gfm_exclusion(input_data_meta, datetime_range=None, asset_key=None):
    """
    Fetch the GFM SAR exclusion mask from EODC STAC and reproject to the SAR grid.
    Same calling pattern as get_wo / get_lcc. Skip from SIENA.py with --skip_gfm_exclusion.
    """
    from gfm_sar_exclusion_mars import get_gfm_sar_exclusion
    return get_gfm_sar_exclusion(
        input_data_meta,
        source="eodc",
        cache_dir=GFM_exclusion_folder,
        datetime_range=datetime_range,
        asset_key=asset_key,
    )

# -----------------------
# Original “_ori” functions (kept)
# -----------------------
def get_dem_ori(input_data_meta):
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    dem_list = glo30list(wgs84_bounds)
    DEM_merge_file = mergelist(dem_list)
    dem_crop = cropfile(DEM_merge_file, wgs84_bounds)
    return read_raster(dem_crop)

def get_HAND30_ori(input_data_meta):
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    hand_list = HAND30list(wgs84_bounds)
    hand_merge_file = mergelist(hand_list)
    hand_crop = cropfile(hand_merge_file, wgs84_bounds)
    return read_raster(hand_crop)

def get_3depdem(input_data_meta):
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    dem_list = DEP3list(wgs84_bounds)
    DEM_merge_file = mergelist(dem_list)
    dem_crop = cropfile(DEM_merge_file, wgs84_bounds)
    return read_raster(dem_crop)

def get_fabdem(input_data_meta):
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    dem_list = FABDEMlist(wgs84_bounds)
    DEM_merge_file = mergelist(dem_list)
    dem_crop = cropfile(DEM_merge_file, wgs84_bounds)
    return read_raster(dem_crop)

def get_lcc_ori(input_data_meta):
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    lcc_list = getLCC(wgs84_bounds)
    lcc_merge_file = mergelist(lcc_list)
    lcc_crop = cropfile(lcc_merge_file, wgs84_bounds)
    return read_raster(lcc_crop)

def get_wo_ori(input_data_meta):
    wgs84_bounds = convert_bounds(input_data_meta['meta']['crs'], input_data_meta['bounds'])
    wo_list = getWO(wgs84_bounds)
    wo_merge_file = mergelist(wo_list)
    wo_crop = cropfile(wo_merge_file, wgs84_bounds)
    return read_raster(wo_crop)

# -----------------------
# Utilities kept, with a small fix in pixel-size calc (pyproj)
# -----------------------
def calculate_pixel_size(raster_data_dict):
    metadata = raster_data_dict['meta']
    transform = metadata['transform']
    pixel_width  = transform[0]
    pixel_height = abs(transform[4])
    crs = metadata['crs']

    pixel_width_m = pixel_width
    pixel_height_m = pixel_height

    if crs.is_geographic:
        bounds = raster_data_dict['bounds']
        top_left_lon = bounds.left
        top_left_lat = bounds.top
        tform = Transformer.from_crs(crs, 'EPSG:3857', always_xy=True)
        top_left_x, top_left_y = tform.transform(top_left_lon, top_left_lat)
        bottom_right_x, bottom_right_y = tform.transform(top_left_lon + pixel_width, top_left_lat - pixel_height)
        pixel_width_m  = abs(bottom_right_x - top_left_x)
        pixel_height_m = abs(bottom_right_y - top_left_y)

    average_pixel_size_m = (pixel_width_m + pixel_height_m) / 2
    pixel_area_m2  = pixel_width_m * pixel_height_m
    pixel_area_km2 = pixel_area_m2 / 1e6
    return pixel_area_km2, average_pixel_size_m

# -----------------------
# Tile list helpers (unchanged API)
# -----------------------
def glo30list(bounds:tuple) -> list:
    if ogr is None:
        print('glo30list: osgeo/ogr not available; using no DEM tiles.')
        return []
    local_idx = _offline_geojson_dem_index()
    if SIENA_OFFLINE:
        if not os.path.isfile(local_idx):
            raise RuntimeError(
                "SIENA_OFFLINE: missing DEM tile index GeoJSON: %s\n"
                "On login: curl -o ... or python check_env.py (saves cop30-2021.geojson into DEM_download)."
                % local_idx
            )
        DEM_GEOJSON = local_idx
    else:
        DEM_GEOJSON = _ensure_index_geojson(local_idx, _INDEX_URL_DEM)
    dataSource = ogr.Open(DEM_GEOJSON)
    if dataSource is None:
        print('glo30list: failed to open DEM GeoJSON (network?); using no DEM tiles.')
        return []
    layer = dataSource.GetLayer()
    extent_geom = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bounds[0], bounds[1])
    ring.AddPoint(bounds[2], bounds[1])
    ring.AddPoint(bounds[2], bounds[3])
    ring.AddPoint(bounds[0], bounds[3])
    ring.AddPoint(bounds[0], bounds[1])
    extent_geom.AddGeometry(ring)
    elements = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if extent_geom.Intersects(geom):
            file_path = feature.GetField('file_path')
            print(file_path)
            elements.append(file_path)
    dataSource = None
    return elements

def DEP3list(bounds:tuple) -> list:
    """Optional local 3DEP DEM index. Not used by the default Copernicus GLO-30 path.

    Set SIENA_3DEP_GEOJSON to a local tile-index GeoJSON if you call this helper.
    """
    if ogr is None:
        return []
    DEM_GEOJSON = os.environ.get("SIENA_3DEP_GEOJSON", "")
    if not DEM_GEOJSON or not os.path.isfile(DEM_GEOJSON):
        return []
    dataSource = ogr.Open(DEM_GEOJSON)
    if dataSource is None:
        return []
    layer = dataSource.GetLayer()
    extent_geom = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bounds[0], bounds[1])
    ring.AddPoint(bounds[2], bounds[1])
    ring.AddPoint(bounds[2], bounds[3])
    ring.AddPoint(bounds[0], bounds[3])
    ring.AddPoint(bounds[0], bounds[1])
    extent_geom.AddGeometry(ring)
    elements = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if extent_geom.Intersects(geom):
            file_path = feature.GetField('file_path')
            print(file_path)
            elements.append(file_path)
    dataSource = None
    return elements

def FABDEMlist(bounds:tuple) -> list:
    """Optional local FABDEM index. Not used by the default Copernicus GLO-30 path.

    Set SIENA_FABDEM_GEOJSON to a local tile-index GeoJSON if you call this helper.
    """
    if ogr is None:
        return []
    DEM_GEOJSON = os.environ.get("SIENA_FABDEM_GEOJSON", "")
    if not DEM_GEOJSON or not os.path.isfile(DEM_GEOJSON):
        return []
    dataSource = ogr.Open(DEM_GEOJSON)
    if dataSource is None:
        return []
    layer = dataSource.GetLayer()
    extent_geom = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bounds[0], bounds[1])
    ring.AddPoint(bounds[2], bounds[1])
    ring.AddPoint(bounds[2], bounds[3])
    ring.AddPoint(bounds[0], bounds[3])
    ring.AddPoint(bounds[0], bounds[1])
    extent_geom.AddGeometry(ring)
    elements = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if extent_geom.Intersects(geom):
            file_path = feature.GetField('file_path')
            print(file_path)
            elements.append(file_path)
    dataSource = None
    return elements

def HAND30list(bounds:tuple) -> list:
    if ogr is None:
        print('HAND30list: osgeo/ogr not available; using no HAND tiles.')
        return []
    local_idx = _offline_geojson_hand_index()
    if SIENA_OFFLINE:
        if not os.path.isfile(local_idx):
            raise RuntimeError(
                "SIENA_OFFLINE: missing HAND index GeoJSON: %s (pre-download on login; see check_env.py)"
                % local_idx
            )
        DEM_GEOJSON = local_idx
    else:
        DEM_GEOJSON = _ensure_index_geojson(local_idx, _INDEX_URL_HAND)
    dataSource = ogr.Open(DEM_GEOJSON)
    if dataSource is None:
        print('HAND30list: failed to open HAND GeoJSON (network?); using no HAND tiles.')
        return []
    layer = dataSource.GetLayer()
    extent_geom = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bounds[0], bounds[1])
    ring.AddPoint(bounds[2], bounds[1])
    ring.AddPoint(bounds[2], bounds[3])
    ring.AddPoint(bounds[0], bounds[3])
    ring.AddPoint(bounds[0], bounds[1])
    extent_geom.AddGeometry(ring)
    elements = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if extent_geom.Intersects(geom):
            file_path = feature.GetField('file_path')
            print(file_path)
            elements.append(file_path)
    dataSource = None
    return elements

def getWO(bounds:tuple) -> list:
    Hcrop_Ex = bounds
    maxLoESA = int(np.ceil(Hcrop_Ex[2] / 10) * 10)
    minLoESA = int(np.floor(Hcrop_Ex[0] / 10) * 10)
    maxLaESA = int(np.ceil(Hcrop_Ex[3] / 10) * 10 + 10)
    minLaESA = int(np.ceil(Hcrop_Ex[1] / 10) * 10)

    if maxLoESA > 0 and minLoESA > 0:
        lons = [str(e) + 'E' for e in range(minLoESA, maxLoESA, 10)]
    elif maxLoESA < 0 and minLoESA < 0:
        lons = [str(w) + 'W' for w in np.absolute(range(minLoESA, maxLoESA, 10))]
    else:
        lons = [str(w) + 'W' for w in np.absolute(range(minLoESA, 0, 10))]
        lons.extend([str(e) + 'E' for e in range(0, maxLoESA, 10)])

    if maxLaESA > 0 and minLaESA > 0:
        lats = [str(n) + 'N' for n in range(minLaESA, maxLaESA, 10)]
    elif maxLaESA < 0 and minLaESA < 0:
        lats = [str(s) + 'S' for s in np.absolute(range(minLaESA, maxLaESA, 10))]
    else:
        lats = [str(s) + 'S' for s in np.absolute(range(minLaESA, 0, 10))]
        lats.extend([str(n) + 'N' for n in range(0, maxLaESA, 10)])

    ESArevision = '1_3_2020'
    ESAdatasets = ['occurrence']
    url_tmpl, file_tmpl, _ = templatesESAwater(ESArevision)

    elements=[]
    for ds_name in ESAdatasets:
        for lon in lons:
            for lat in lats:
                filename = file_tmpl.format(ds=ds_name, lon=lon, lat=lat)
                url = url_tmpl.format(ds=ds_name, file=filename)
                print(url)
                elements.append(url)
    return elements

def getGPLCC(bounds:tuple) -> list:
    Hcrop_Ex = bounds
    LCCmainURL = "http://data.ess.tsinghua.edu.cn/data/fromglc10_2017v01/fromglc10v01"
    LCCfileregex = "_LA_LO.tif"
    maxLoLCC = np.ceil(Hcrop_Ex[2])
    minLoLCC = np.floor(Hcrop_Ex[0])
    maxLaLCC = np.ceil(Hcrop_Ex[3])
    minLaLCC = np.floor(Hcrop_Ex[1])
    Lo_arange = np.arange(minLoLCC - 1, maxLoLCC + 1, 1)
    La_arange = np.arange(minLaLCC - 1, maxLaLCC + 1, 1)
    Lo_arangeGPLCC = Lo_arange[Lo_arange % 2 == 0]
    La_arangeGPLCC = La_arange[La_arange % 2 == 0]
    elements = []
    for Lo in Lo_arangeGPLCC:
        for La in La_arangeGPLCC:
            LCCfileName = LCCfileregex.replace("LO", str(int(Lo)))
            LCCfileName = LCCfileName.replace("LA", str(int(La)))
            url = LCCmainURL + LCCfileName
            print(url)
            elements.append(url)
    return elements

def getLCC(bounds:tuple) -> list:
    if ogr is None:
        print('getLCC: osgeo/ogr not available; using no LCC tiles.')
        return []
    Hcrop_Ex = bounds
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    local_idx = _offline_geojson_lcc_index()
    if SIENA_OFFLINE:
        if not os.path.isfile(local_idx):
            raise RuntimeError(
                "SIENA_OFFLINE: missing ESA WorldCover grid GeoJSON: %s (pre-download on login; see check_env.py)"
                % local_idx
            )
        DEM_GEOJSON = local_idx
    else:
        DEM_GEOJSON = _ensure_index_geojson(local_idx, _INDEX_URL_LCC_GRID)
    dataSource = ogr.Open(DEM_GEOJSON)
    if dataSource is None:
        print('getLCC: failed to open LCC grid GeoJSON (network/403?); using no LCC tiles.')
        return []
    layer = dataSource.GetLayer()
    extent_geom = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(Hcrop_Ex[0], Hcrop_Ex[1])
    ring.AddPoint(Hcrop_Ex[2], Hcrop_Ex[1])
    ring.AddPoint(Hcrop_Ex[2], Hcrop_Ex[3])
    ring.AddPoint(Hcrop_Ex[0], Hcrop_Ex[3])
    ring.AddPoint(Hcrop_Ex[0], Hcrop_Ex[1])
    extent_geom.AddGeometry(ring)
    elements = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if extent_geom.Intersects(geom):
            tile = feature.GetField('ll_tile')
            file_path = f"{s3_url_prefix}/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
            print(file_path)
            elements.append(file_path)
    return elements

def templatesESAwater(revision):
    REVISIONS = ['1_0', '1_1', '1_1_2019', '1_3_2020']
    v10, v11, v11_2019, v13_2020 = REVISIONS
    url_tmpl = 'http://storage.googleapis.com/global-surface-water/downloads'
    file_tmpl = '{ds}_{lon}_{lat}'
    if revision == v10:
        padding = 15
    elif revision == v11:
        url_tmpl += '2'
        file_tmpl += '_v' + v11
        padding = 20
    elif revision == v11_2019:
        url_tmpl += '2019v2'
        file_tmpl += 'v' + v11_2019
        padding = 24
    elif revision == v13_2020:
        url_tmpl += '2020'
        file_tmpl += 'v' + v13_2020
        padding = 24
    url_tmpl += '/{ds}/{file}'
    file_tmpl += '.tif'
    return (url_tmpl, file_tmpl, padding)