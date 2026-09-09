"""
Load SAR input and attach ancillary layers on the same grid.

Supports GeoTIFF / zip and NetCDF (.nc). NetCDF field names are described in
README_NRCS_NC.md. Ancillary rasters (water occurrence, land cover, DEM, HAND,
optional GFM exclusion) are fetched from public cloud sources by default and
cached on disk; pass --ancillary-download-folder or set SIENA_OFFLINE=1 to use
only local tiles. See SIENA.py and README.md.
"""
import os
import numpy as np
import re
from io import BytesIO
import zipfile
import requests
from scipy import ndimage
import netCDF4
import rasterio
from rasterio.transform import Affine

from utils import tim
from utils import read_raster, create_raster_data_dict
from ancillarydata_merge import (
    get_wo,
    get_lcc,
    get_dem,
    get_HAND30,
    get_gfm_exclusion,
)


def _read_nc_siena(nc_path):
    """
    Read sigma and geographic bounds from a NetCDF SAR file.

    Returns (sigma, left, right, bottom, top, flip_x, flip_y, lat, lon).
    sigma is (ny, nx) float32. lat, lon are 2D when present, else None.
    flip_x/flip_y are used only when geocoding from corner attributes.
    See README_NRCS_NC.md for expected variables.
    """
    nc_path = os.path.abspath(nc_path)
    if not os.path.isfile(nc_path):
        raise FileNotFoundError(nc_path)
    with netCDF4.Dataset(nc_path, "r") as nc:
        if "sigma" not in nc.variables:
            raise ValueError("NetCDF missing variable 'sigma'")
        sigma = np.array(nc.variables["sigma"][:], dtype=np.float32)
        flip_x = False
        flip_y = False
        lat = None
        lon = None
        if "latitude" in nc.variables and "longitude" in nc.variables:
            lat = np.array(nc.variables["latitude"][:])
            lon = np.array(nc.variables["longitude"][:])
            left = float(np.nanmin(lon))
            right = float(np.nanmax(lon))
            bottom = float(np.nanmin(lat))
            top = float(np.nanmax(lat))
            lon_col0 = np.nanmean(lon[:, 0])
            lon_col_last = np.nanmean(lon[:, -1])
            flip_x = lon_col0 > lon_col_last
            lat_row0 = np.nanmean(lat[0, :])
            lat_row_last = np.nanmean(lat[-1, :])
            flip_y = lat_row0 < lat_row_last
        else:
            left = float(getattr(nc, "ul_outer_lon", getattr(nc, "ll_outer_lon", -180)))
            right = float(getattr(nc, "ur_outer_lon", getattr(nc, "lr_outer_lon", 180)))
            bottom = float(getattr(nc, "ll_outer_lat", getattr(nc, "ul_outer_lat", -90)))
            top = float(getattr(nc, "ul_outer_lat", getattr(nc, "ur_outer_lat", 90)))
            left, right = min(left, right), max(left, right)
            bottom, top = min(bottom, top), max(bottom, top)
            ul = getattr(nc, "ul_outer_lon", None)
            ur = getattr(nc, "ur_outer_lon", None)
            if ul is not None and ur is not None:
                flip_x = float(ul) > float(ur)
        return sigma, left, right, bottom, top, flip_x, flip_y, lat, lon


def _geocode_nc_exact_resample(sigma, lat, lon, left, right, bottom, top):
    """
    Forward-map one NetCDF array onto a regular WGS84 grid.

    Each source pixel contributes to the target cell its (lon, lat) falls in;
    overlapping contributions are averaged. No inverse interpolation.
    """
    from rasterio.transform import from_bounds, rowcol

    ny, nx = sigma.shape
    transform = from_bounds(left, bottom, right, top, nx, ny)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(sigma)
    if not np.any(valid):
        raise RuntimeError("No valid (lat, lon, sigma) points in NC for geocoding.")
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    sigma_flat = sigma.ravel()
    valid_flat = valid.ravel()
    rows, cols = rowcol(transform, lon_flat[valid_flat], lat_flat[valid_flat])
    r = np.rint(rows).astype(np.int32)
    c = np.rint(cols).astype(np.int32)
    in_bounds = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    r = r[in_bounds]
    c = c[in_bounds]
    sigma_contrib = sigma_flat[valid_flat][in_bounds]
    sigma_reg_sum = np.zeros((ny, nx), dtype=np.float64)
    sigma_reg_count = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(sigma_reg_sum, (r, c), sigma_contrib)
    np.add.at(sigma_reg_count, (r, c), 1.0)
    sigma_reg = np.where(
        sigma_reg_count > 0,
        sigma_reg_sum / sigma_reg_count,
        0.0,
    ).astype(np.float32)
    bounds = (left, bottom, right, top)
    return sigma_reg, transform, bounds


def _geocode_nc_forward_one(arr, lat, lon, transform, ny, nx):
    """Forward-map one optional NetCDF field (incidence, mask, variance) onto the same grid as sigma."""
    from rasterio.transform import rowcol
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(arr)
    if not np.any(valid):
        return np.zeros((ny, nx), dtype=np.float32)
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    arr_flat = np.asarray(arr, dtype=np.float64).ravel()
    valid_flat = valid.ravel()
    rows, cols = rowcol(transform, lon_flat[valid_flat], lat_flat[valid_flat])
    r = np.rint(rows).astype(np.int32)
    c = np.rint(cols).astype(np.int32)
    in_bounds = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    r, c = r[in_bounds], c[in_bounds]
    arr_contrib = arr_flat[valid_flat][in_bounds]
    out_sum = np.zeros((ny, nx), dtype=np.float64)
    out_count = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(out_sum, (r, c), arr_contrib)
    np.add.at(out_count, (r, c), 1.0)
    out = np.where(out_count > 0, out_sum / out_count, 0.0).astype(np.float32)
    return out


def _read_nc_optional_fields(nc_path, lat, lon, transform, ny, nx):
    """Read optional NetCDF variables incid, mask, variance and place them on the sigma grid."""
    result = {"incid": None, "nc_mask": None, "nc_variance": None}
    with netCDF4.Dataset(nc_path, "r") as nc:
        if "incid" in nc.variables:
            inc = np.array(nc.variables["incid"][:], dtype=np.float32)
            if inc.shape == lat.shape:
                result["incid"] = _geocode_nc_forward_one(inc, lat, lon, transform, ny, nx)
        if "mask" in nc.variables:
            mask = np.array(nc.variables["mask"][:], dtype=np.float32)
            if mask.shape == lat.shape:
                result["nc_mask"] = _geocode_nc_forward_one(mask, lat, lon, transform, ny, nx)
        if "variance" in nc.variables:
            var = np.array(nc.variables["variance"][:], dtype=np.float32)
            if var.shape == lat.shape:
                result["nc_variance"] = _geocode_nc_forward_one(var, lat, lon, transform, ny, nx)
    return result


def _geocode_nc_via_gcps(sigma, lat, lon, nx, ny):
    """
    Geocode sigma from per-pixel lat/lon using ground-control points.
    Prefer _geocode_nc_exact_resample when 2D lat/lon are available.
    """
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import from_gcps

    step = max(1, min(nx, ny) // 25)
    gcps = []
    for r in range(0, ny, step):
        for c in range(0, nx, step):
            loni = lon[r, c]
            lati = lat[r, c]
            if np.isfinite(loni) and np.isfinite(lati):
                gcps.append(GroundControlPoint(row=r, col=c, x=float(loni), y=float(lati)))
    gcp_pixels = {(g.row, g.col) for g in gcps}
    for r, c in [(0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1)]:
        if (r, c) not in gcp_pixels and np.isfinite(lon[r, c]) and np.isfinite(lat[r, c]):
            gcps.append(GroundControlPoint(row=r, col=c, x=float(lon[r, c]), y=float(lat[r, c])))
    if len(gcps) < 4:
        raise RuntimeError("Not enough valid GCPs from latitude/longitude (need at least 4).")
    gcp_crs = "EPSG:4326"
    src_transform = from_gcps(gcps)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        gcp_crs, gcp_crs, nx, ny, gcps=gcps
    )
    sigma_reg = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=sigma,
        destination=sigma_reg,
        src_transform=src_transform,
        src_crs=gcp_crs,
        dst_transform=dst_transform,
        dst_crs=gcp_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    sigma_reg = np.where(np.isfinite(sigma_reg), sigma_reg, 0.0).astype(np.float32)
    west = dst_transform.c
    east = dst_transform.c + dst_width * dst_transform.a
    north = dst_transform.f
    south = dst_transform.f + dst_height * dst_transform.e
    bounds = (west, south, east, north)
    return sigma_reg, dst_transform, bounds


def tosigmma(arr):
    """Convert amplitude-like values to a linear sigma-style scale when the filename contains 'amp'."""
    return np.power(10, (2 * np.log10(arr) - 8.3))


def readfileio(file):
    """Read band 1 of a GeoTIFF as float32."""
    with open(file, "rb") as _:
        pass  # ensure local file
    import rasterio
    with rasterio.open(file) as src:
        img = np.array(src.read(1), dtype=np.float32)
    return img


def readfileio_multiband(file):
    """Read all bands of a GeoTIFF as float32."""
    import rasterio
    with rasterio.open(file) as src:
        img = np.array(src.read(), dtype=np.float32)
    return img


def filepathreader(ds, varnames, infilepath):
    """Read co-pol (img_lp) and cross-pol (img_cp). One file and one band: copy co-pol into img_cp."""
    import rasterio
    with rasterio.open(infilepath[0]) as src:
        num_bands = src.count
    if num_bands >= 2:
        with rasterio.open(infilepath[0]) as src:
            setattr(ds, varnames[0], np.array(src.read(1), dtype=np.float32))
            setattr(ds, varnames[1], np.array(src.read(2), dtype=np.float32))
            if len(varnames) > 2:
                setattr(ds, varnames[2], np.array(src.read(3), dtype=np.float32) if num_bands >= 3 else None)
    elif len(infilepath) >= 2:
        for idx, filename in enumerate(infilepath[:2]):
            setattr(ds, varnames[idx], readfileio(filename))
            if "amp" in filename.lower():
                setattr(ds, varnames[idx], tosigmma(getattr(ds, varnames[idx])))
    else:
        # Single file, single band: co-pol only; duplicate for img_cp so dual-pol pipeline works
        setattr(ds, varnames[0], readfileio(infilepath[0]))
        if "amp" in infilepath[0].lower():
            setattr(ds, varnames[0], tosigmma(getattr(ds, varnames[0])))
        setattr(ds, varnames[1], np.array(getattr(ds, varnames[0]), dtype=np.float32, copy=True))
    return ds


def get_src(filepath, args):
    """Attach rasterio dataset metadata from the first input file."""
    import rasterio
    with rasterio.open(filepath) as src:
        args.ref_src = src
    return args


NRCS_NODATA_DB = -50.0

# Incidence normalization exponent (sigma / cos(inc_rad)^INCID_NORM_K)
INCID_NORM_K = 0.5


def _apply_incid_normalization(ds, args):
    """
    When the NetCDF has a valid incidence grid and --use_incid_normalization is set,
    flatten range dependence: sigma ← sigma / cos(incidence)^k.
    """
    if not getattr(ds, "from_nc", False):
        return
    if not getattr(args, "use_incid_normalization", False):
        return
    inc = getattr(ds, "inc", None)
    if inc is None:
        return
    inc = np.asarray(inc, dtype=np.float64)
    # Valid incidence: finite and in reasonable range (e.g. 0--70 deg)
    valid_inc = np.isfinite(inc) & (inc >= 0) & (inc <= 90)
    if not np.any(valid_inc):
        return
    inc_rad = np.deg2rad(np.where(valid_inc, inc, 0.0))
    cos_inc = np.cos(inc_rad)
    # Avoid div by zero; only normalize where cos > epsilon
    eps = 1e-6
    safe = cos_inc > eps
    if not np.any(safe):
        return
    factor = np.where(safe, np.power(np.maximum(cos_inc, eps), -INCID_NORM_K), 1.0)
    ds.img_lp = np.where(safe & (ds.img_lp > 0), ds.img_lp * factor, ds.img_lp).astype(np.float32)
    ds.img_cp = np.where(safe & (ds.img_cp > 0), ds.img_cp * factor, ds.img_cp).astype(np.float32)
    print("Incidence normalization applied (sigma / cos(inc)^%.1f)." % INCID_NORM_K)


def db_to_linear_sigma0(ds):
    """Convert dB backscatter to linear sigma0 when the GeoTIFF name contains 'nrcs'."""
    ds.img_lp = np.where(
        ds.img_lp <= NRCS_NODATA_DB,
        0.0,
        np.where(np.isfinite(ds.img_lp), 10.0 ** (ds.img_lp / 10.0), 0.0),
    )
    ds.img_cp = np.where(
        ds.img_cp <= NRCS_NODATA_DB,
        0.0,
        np.where(np.isfinite(ds.img_cp), 10.0 ** (ds.img_cp / 10.0), 0.0),
    )
    return ds


def set_nan(ds, args):
    """Zero invalid pixels (nodata, optional NetCDF mask, optional edge erosion)."""
    nodata = getattr(args.ref_src, "nodata", None)
    if nodata is not None:
        ds.img_lp = np.where(ds.img_lp == nodata, 0, ds.img_lp)
        ds.img_cp = np.where(ds.img_cp == nodata, 0, ds.img_cp)
    # B) NC mask for valid pixels: when from .nc and nc_mask exists with usable values, zero out where mask is 0/nan (and optionally water)
    if getattr(ds, "from_nc", False):
        nc_mask = getattr(ds, "nc_mask", None)
        if nc_mask is not None:
            nc_mask = np.asarray(nc_mask, dtype=np.float64)
            # NRCS: positive=land, negative=water; 0 = no data.
            # By default: valid = finite and != 0 (land and water kept). With --nc_mask_water_as_nodata: valid = land only.
            water_as_nodata = getattr(args, "nc_mask_water_as_nodata", False)
            if water_as_nodata:
                valid_from_mask = np.isfinite(nc_mask) & (nc_mask > 0)
                msg = "Valid pixels from NC mask applied (zeroed where mask is 0, non-finite, or water/negative)."
            else:
                valid_from_mask = np.isfinite(nc_mask) & (nc_mask != 0)
                msg = "Valid pixels from NC mask applied (zeroed where mask is 0 or non-finite)."
            if np.any(valid_from_mask):
                ds.img_lp = np.where(valid_from_mask, ds.img_lp, 0).astype(np.float32)
                ds.img_cp = np.where(valid_from_mask, ds.img_cp, 0).astype(np.float32)
                print(msg)
    brd_pixels = getattr(args, "brd_erosion_pixels", 0)
    if brd_pixels > 0:
        print("BRD noise: eroding valid sigma by %d pixels (--brd_erosion_pixels)" % brd_pixels)
        valid_mask = np.where(ds.img_lp > 0, 1, 0).astype(np.uint8)
        valid_BRD_mask = ndimage.binary_erosion(valid_mask, iterations=brd_pixels).astype(np.uint8)
        ds.img_lp = np.where(valid_BRD_mask == 1, ds.img_lp, 0).astype(np.float32)
        ds.img_cp = np.where(valid_BRD_mask == 1, ds.img_cp, 0).astype(np.float32)
    return ds


def _remap_lcc_to_esa_world_cover(arr):
    """
    Optional remap: only use if your LCC source is NOT ESA World Cover.
    ESA World Cover codes (unchanged): 10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland,
    50 Built-up, 60 Bare/sparse vegetation, 70 Snow and ice, 80 Permanent water bodies,
    90 Herbaceous wetland, 95 Mangroves, 100 Moss and lichen.
    """
    return np.asarray(arr, dtype=np.float32)


def ESALCC2GPLCC(arr):
    """Deprecated: was remapping to a different (GPLCC) scheme. SIENA now uses ESA World Cover codes as-is."""
    return np.asarray(arr, dtype=np.float32)


def _read_band_from_zip(zip_file, file_name):
    import rasterio
    with zip_file.open(file_name) as tiff_file:
        with rasterio.open(BytesIO(tiff_file.read())) as src:
            arr = np.array(src.read(1), dtype=np.float32)
    return arr


def retrieveRTC(url, args, ds):
    """Download an HTTPS zip of RTC-style GeoTIFFs and read co-pol / cross-pol bands."""
    args.tempfiles = []
    ds.img_lp = None
    ds.img_cp = None
    response = requests.get(url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    vv_path, vh_path = None, None
    for file_name in zip_file.namelist():
        if ("VV" in file_name or "HH" in file_name) and ".xml" not in file_name:
            print("reading VV or HH", file_name)
            arr = _read_band_from_zip(zip_file, file_name)
            ds.img_lp = arr
            vv_path = file_name
        if ("VH" in file_name or "HV" in file_name) and ".xml" not in file_name:
            print("reading VH or HV", file_name)
            arr = _read_band_from_zip(zip_file, file_name)
            ds.img_cp = arr
            vh_path = file_name
        if "inc_map" in file_name and ".xml" not in file_name:
            print("reading incidence angle", file_name)
            ds.inc = _read_band_from_zip(zip_file, file_name)
    # Single-pol: if only co-pol found, duplicate for img_cp
    if not hasattr(ds, "img_cp") or ds.img_cp is None:
        ds.img_cp = np.array(ds.img_lp, dtype=np.float32, copy=True)
    # ref_src from first VV/HH for metadata
    with zip_file.open(vv_path or vh_path) as tiff_file:
        import rasterio
        with rasterio.open(BytesIO(tiff_file.read())) as src:
            args.ref_src = src
    return args, ds


@tim
def Load_Data(args):
    """
    Initialization for one granule: load SAR, geocode if NetCDF, attach ancillary.

    Ancillary layers are downloaded from public cloud URLs into the cache folder
    unless they are already on disk (or SIENA_OFFLINE=1).
    """
    from urllib.parse import urlparse
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.io import MemoryFile

    filepath = args.filepath
    args.varnames = ["img_lp", "img_cp", "inc"]
    class c:
        pass
    ds = c()
    ds.inc = None
    ds.from_nc = False
    SAR_img_d_m = None

    if urlparse(filepath[0]).scheme == "https" and ".tif" not in filepath[0]:
        print("Input with ASF url " + filepath[0])
        args, ds = retrieveRTC(filepath[0], args, ds)
    else:
        infilepath = filepath
        if filepath[0].lower().endswith(".nc"):
            # NRCS NetCDF: sigma already linear; geocode via GCPs when 2D lat/lon exist (README_NRCS_NC.md)
            print("Input NRCS NetCDF: " + filepath[0])
            sigma_raw, left, right, bottom, top, flip_x, flip_y, lat_2d, lon_2d = _read_nc_siena(filepath[0])
            # Treat tiny magnitudes as no-data: anything strictly below 1e-6 goes to NaN so it is
            # excluded from geocoding / resampling and downstream stats (valid mask uses np.isfinite).
            tiny_mask = np.isfinite(sigma_raw) & (np.abs(sigma_raw) < 1e-6)
            if np.any(tiny_mask):
                sigma_raw = sigma_raw.astype(np.float32, copy=True)
                sigma_raw[tiny_mask] = np.nan
            # Keep an untouched copy of the NC grid (after tiny->NaN) for optional debug output
            sigma = np.array(sigma_raw, dtype=np.float32, copy=True)
            ny, nx = sigma.shape
            if lat_2d is not None and lon_2d is not None:
                # Exact geocoding: resample sigma onto regular grid by interpolating at each (lon,lat)
                # so ancillary reprojected to the same grid matches pixel-perfect (no GCP affine residual).
                print("Geocoding: exact resample onto regular grid from 2D lat/lon (pixel-perfect with ancillary).")
                sigma, transform, bounds = _geocode_nc_exact_resample(
                    sigma, lat_2d, lon_2d, left, right, bottom, top
                )
                ny, nx = sigma.shape
            else:
                # Fallback: from_bounds + flip (may misalign if grid is non-affine)
                if flip_x:
                    sigma = np.ascontiguousarray(sigma[:, ::-1])
                if flip_y:
                    sigma = np.ascontiguousarray(sigma[::-1, :])
                transform = from_bounds(left, bottom, right, top, nx, ny)
                bounds = (left, bottom, right, top)
            meta = {
                "driver": "GTiff",
                "dtype": "float32",
                "width": nx,
                "height": ny,
                "count": 1,
                "crs": "EPSG:4326",
                "transform": transform,
                "nodata": None,
            }
            output_intermediate = getattr(args, "output_intermediate", False)
            sigma_arr = np.asarray(sigma, dtype=np.float32)
            ds.img_lp = sigma_arr
            ds.img_cp = np.array(ds.img_lp, dtype=np.float32, copy=True)
            ds.from_nc = True
            args.tempfiles = getattr(args, "tempfiles", [])

            if output_intermediate:
                out_dir = getattr(args, "dirOut", None)
                if not out_dir:
                    import tempfile
                    out_dir = tempfile.gettempdir()
                os.makedirs(out_dir, exist_ok=True)
                # Raw NC sigma grid (no reprojection/resample, no CRS) for value-range inspection
                raw_h, raw_w = sigma_raw.shape
                raw_meta = {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "width": raw_w,
                    "height": raw_h,
                    "count": 1,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
                    "nodata": None,
                }
                out_tif_raw = os.path.join(out_dir, "sar_from_nc_raw_grid.tif")
                with rasterio.open(out_tif_raw, "w", **raw_meta) as dst:
                    dst.write(np.asarray(sigma_raw, dtype=np.float32), 1)
                print("Exported raw NC sigma grid (no reprojection): %s" % out_tif_raw)

                out_tif = os.path.join(out_dir, "sar_from_nc.tif")
                with rasterio.open(out_tif, "w", **meta) as dst:
                    dst.write(sigma_arr, 1)
                NODATA_DB = -50.0
                sigma_db = np.where(
                    np.isfinite(sigma) & (sigma > 0),
                    10.0 * np.log10(np.maximum(sigma, 1e-12)),
                    NODATA_DB,
                ).astype(np.float32)
                meta_db = meta.copy()
                meta_db["nodata"] = NODATA_DB
                out_tif_db = os.path.join(out_dir, "sar_from_nc_dB.tif")
                with rasterio.open(out_tif_db, "w", **meta_db) as dst:
                    dst.write(sigma_db, 1)
                print("Exported dB image: %s" % out_tif_db)
                args.ref_src = rasterio.open(out_tif)
            else:
                buf = MemoryFile()
                with buf.open(**meta) as dst:
                    dst.write(sigma_arr, 1)
                args.ref_src = buf.open()

            # Optional NC fields (incid, mask, variance) on same grid; only write mask TIFs when output_intermediate
            opt = _read_nc_optional_fields(filepath[0], lat_2d, lon_2d, transform, ny, nx)
            if opt["incid"] is not None:
                ds.inc = opt["incid"]
            if opt["nc_mask"] is not None:
                ds.nc_mask = opt["nc_mask"]
                if output_intermediate:
                    import tempfile
                    out_dir = getattr(args, "dirOut", None) or tempfile.gettempdir()
                    os.makedirs(out_dir, exist_ok=True)
                    out_mask_tif = os.path.join(out_dir, "mask_from_nc.tif")
                    with rasterio.open(out_mask_tif, "w", **meta) as dst:
                        dst.write(np.asarray(opt["nc_mask"], dtype=np.float32), 1)
                    print("Exported NC mask: %s (NRCS: positive=land, negative=water, 0=no data)" % out_mask_tif)
                    m = np.asarray(opt["nc_mask"], dtype=np.float64)
                    cls = np.zeros(m.shape, dtype=np.uint8)
                    cls[np.isfinite(m) & (m > 0)] = 1
                    cls[np.isfinite(m) & (m < 0)] = 2
                    meta_3 = {k: v for k, v in meta.items()}
                    meta_3["dtype"] = "uint8"
                    meta_3["nodata"] = 0
                    out_3class = os.path.join(out_dir, "mask_from_nc_3class.tif")
                    with rasterio.open(out_3class, "w", **meta_3) as dst:
                        dst.write(cls, 1)
                    print("Exported 3-class mask: %s (0=no data, 1=land, 2=water)" % out_3class)
            if opt["nc_variance"] is not None:
                ds.nc_variance = opt["nc_variance"]
            SAR_img_d_m = create_raster_data_dict(
                sigma_arr, meta.copy(), bounds
            )
        else:
            args = get_src(filepath[0], args)
            args.tempfiles = []
            ds = filepathreader(ds, args.varnames, infilepath)
            SAR_img_d_m = None

    args.refmeta = args.ref_src.meta.copy()
    args.refmeta.update(count=1)

    if "nrcs" in os.path.basename(args.filepath[0]).lower() and not getattr(ds, "from_nc", False):
        ds = db_to_linear_sigma0(ds)
    # A) Incidence normalization (when .nc has incid and --use_incid_normalization); skip if no valid inc
    _apply_incid_normalization(ds, args)
    ds = set_nan(ds, args)

    if SAR_img_d_m is None:
        SAR_img_d_m = read_raster(args.filepath[0])

    if getattr(args, "MODE", None) != "desert":
        ds.WO_d_m = get_wo(SAR_img_d_m)
        ds.water_occurrence = np.asarray(ds.WO_d_m["arr"], dtype=np.float32)

        ds.LCC_d_m = get_lcc(SAR_img_d_m)
        ds.land_cover_raw = np.asarray(ds.LCC_d_m["arr"], dtype=np.float32)
        # Keep ESA World Cover codes as-is (10, 20, ..., 50 Built-up, 60 Bare, 80 Permanent water, etc.)
        ds.land_cover = np.asarray(ds.land_cover_raw, dtype=np.float32)

        ds.DEM_d_m = get_dem(SAR_img_d_m)
        ds.HAND_d_m = get_HAND30(SAR_img_d_m)

        skip_gfm = getattr(args, "skip_gfm_exclusion", False)
        if skip_gfm or os.environ.get("SKIP_GFM_EXCLUSION", "").strip() == "1":
            _h, _w = SAR_img_d_m["meta"]["height"], SAR_img_d_m["meta"]["width"]
            _zero = np.zeros((_h, _w), dtype=np.float32)
            ds.GFM_exclusion_d_m = create_raster_data_dict(
                _zero, SAR_img_d_m["meta"].copy(), SAR_img_d_m["bounds"]
            )
            ds.gfm_sar_exclusion = ds.GFM_exclusion_d_m["arr"]
            print("GFM exclusion: skipped")
        else:
            try:
                print("Fetching GFM SAR exclusion (EODC STAC)...")
                ds.GFM_exclusion_d_m = get_gfm_exclusion(SAR_img_d_m)
                ds.gfm_sar_exclusion = np.asarray(ds.GFM_exclusion_d_m["arr"], dtype=np.float32)
                print("GFM SAR exclusion done.")
            except (RuntimeError, ImportError) as e:
                print("GFM exclusion unavailable (%s); using zero mask." % e)
                _h, _w = SAR_img_d_m["meta"]["height"], SAR_img_d_m["meta"]["width"]
                _zero = np.zeros((_h, _w), dtype=np.float32)
                ds.GFM_exclusion_d_m = create_raster_data_dict(
                    _zero, SAR_img_d_m["meta"].copy(), SAR_img_d_m["bounds"]
                )
                ds.gfm_sar_exclusion = ds.GFM_exclusion_d_m["arr"]

    # Always treat as co-pol only downstream: ignore cross-pol by mirroring img_lp into img_cp.
    ds.img_cp = np.array(ds.img_lp, dtype=np.float32, copy=True)

    # Keep ds.inc when from .nc (optional incidence normalization); remove when from zip/RTC if unused
    if hasattr(ds, "inc") and not getattr(ds, "from_nc", False):
        del ds.inc
    return args, ds
