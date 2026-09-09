#!/usr/bin/env python3
"""
Check that Python packages import, ancillary cache folders exist, and public
ancillary URLs are reachable. Run from this directory with no arguments:

  python check_env.py
"""
import os
import sys

# Use script directory so imports find Load_Data, utils, ancillarydata_merge
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Same env defaults as SIENA / ancillarydata_merge
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.json,.geojson")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "1048576")

def main():
    errors = []

    # 1) Required modules
    # (import_name, conda-forge package name)
    required = [
        ("numpy", "numpy"),
        ("rasterio", "rasterio"),
        ("osgeo", "gdal"),
        ("scipy", "scipy"),
        ("requests", "requests"),
        ("shapely", "shapely"),
        ("pyproj", "pyproj"),
        ("psutil", "psutil"),
        ("pandas", "pandas"),
        ("cv2", "opencv"),
        ("netCDF4", "netcdf4"),
        ("matplotlib", "matplotlib"),
        ("pystac_client", "pystac-client"),
    ]
    for mod, conda_pkg in required:
        try:
            __import__(mod)
        except ImportError as e:
            errors.append(
                "Missing module '%s': conda install -c conda-forge %s (%s)"
                % (mod, conda_pkg, e)
            )

    if errors:
        print("FAIL: Missing required modules:")
        for e in errors:
            print("  ", e)
        return 1

    print("OK: Required modules importable.")

    # 2) Ancillary folder layout (same as ancillarydata_merge)
    BASE_ANC_ROOT = os.environ.get("ANCILLARY_ROOT", _script_dir)
    folders = [
        os.path.join(BASE_ANC_ROOT, "ancillary_download", "GSW_download"),
        os.path.join(BASE_ANC_ROOT, "ancillary_download", "LCC_download"),
        os.path.join(BASE_ANC_ROOT, "ancillary_download", "DEM_download"),
        os.path.join(BASE_ANC_ROOT, "ancillary_download", "HAND30_download"),
        os.path.join(BASE_ANC_ROOT, "ancillary_download", "GFM_exclusion_download"),
    ]
    for d in folders:
        try:
            os.makedirs(d, exist_ok=True)
            if not os.path.isdir(d):
                errors.append("Could not create folder: %s" % d)
        except OSError as e:
            errors.append("Permission/error creating %s: %s" % (d, e))

    if errors:
        print("FAIL: Ancillary folders:")
        for e in errors:
            print("  ", e)
        return 1

    print("OK: Ancillary folders created (or already exist).")

    # 3) Check reachability of every ancillary data source (and download index files where applicable)
    import requests
    REQ_TIMEOUT = 30
    ancillary_checks = [
        {
            "name": "DEM (Copernicus 30m)",
            "url": "https://asf-dem-west.s3.amazonaws.com/v2/cop30-2021.geojson",
            "method": "GET",
            "save_subdir": "DEM_download",
            "save_filename": "cop30-2021.geojson",
        },
        {
            "name": "HAND30 (GLO-30 HAND)",
            "url": "https://glo-30-hand.s3.amazonaws.com/v1/2021/glo-30-hand.geojson",
            "method": "GET",
            "save_subdir": "HAND30_download",
            "save_filename": "glo-30-hand.geojson",
        },
        {
            "name": "LCC (ESA World Cover)",
            "url": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/esa_worldcover_2020_grid.geojson",
            "method": "GET",
            "save_subdir": "LCC_download",
            "save_filename": "esa_worldcover_2020_grid.geojson",
        },
        {
            "name": "GSW (Google Surface Water)",
            "url": "http://storage.googleapis.com/global-surface-water/downloads2020/occurrence/occurrence_0E_0Nv1_3_2020.tif",
            "method": "HEAD",
        },
        {
            "name": "GFM (EODC STAC API)",
            "url": "https://stac.eodc.eu/api/v1",
            "method": "GET",
        },
    ]

    for check in ancillary_checks:
        name = check["name"]
        url = check["url"]
        method = check.get("method", "GET")
        print("Checking %s ..." % name, end=" ", flush=True)
        try:
            if method == "HEAD":
                r = requests.head(url, timeout=REQ_TIMEOUT, allow_redirects=True)
            else:
                r = requests.get(url, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            if check.get("save_subdir") and method == "GET":
                folder = os.path.join(BASE_ANC_ROOT, "ancillary_download", check["save_subdir"])
                save_path = os.path.join(folder, check["save_filename"])
                with open(save_path, "wb") as f:
                    f.write(r.content)
                if os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
                    print("OK (downloaded %s)" % check["save_filename"])
                else:
                    print("OK (reachable, but save failed or empty)")
                    errors.append("%s: file not saved or empty" % name)
            else:
                print("OK")
        except requests.RequestException as e:
            print("FAIL: %s" % e)
            errors.append("%s: %s" % (name, e))
        except OSError as e:
            print("FAIL: %s" % e)
            errors.append("%s (write): %s" % (name, e))

    if errors:
        print("FAIL: Some ancillary sources unreachable or write failed:")
        for e in errors:
            print("  ", e)
        return 1

    print("OK: All ancillary sources reachable (and index files downloaded where applicable).")

    # 4) Import SIENA chain (no optional try/import in package code)
    try:
        from utils import reproject_clip_direct  # noqa: F401
        from Load_Data import Load_Data  # noqa: F401
        from ancillarydata_merge import get_wo, get_lcc, get_dem, get_HAND30  # noqa: F401
        import gfm_sar_exclusion_mars  # noqa: F401 — pulls pystac_client (GFM / EODC path)
        print("OK: SIENA module chain importable (utils, Load_Data, ancillarydata_merge, gfm).")
    except ImportError as e:
        print("FAIL: SIENA module chain import failed: %s" % e)
        return 1

    print("Environment check passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
