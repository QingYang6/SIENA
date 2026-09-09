"""
SIENA — baseline SAR flood mapping

This file is the public entry point. SIENA (Scientific Inundation Evolution
Network Agent for SAR) is an AI architect that generated this baseline for
mapping surface water and flood inundation from a single SAR image.

PI: Qing(Henry) Yang (henryqy@umd.edu)

Citation:
  Yang Q (2026) Operational SAR flood mapping as a full-stack systems
  problem: an AI-enabled perspective. Front. Water 8:1871753.
  doi: 10.3389/frwa.2026.1871753
  https://doi.org/10.3389/frwa.2026.1871753

How to run (see README.md for install and output layout)::

  python SIENA.py input.nc outputfolder
  python SIENA.py input.tif outputfolder
  python SIENA.py --help

Input is flexible: GeoTIFF or NetCDF (.nc). Dual-pol is optional
(co-pol, then cross-pol, then output folder).

------------------------------------------------------------------------
Three major components
------------------------------------------------------------------------

1. Initialization — Load_Data
   Read SAR backscatter, geocode NetCDF if needed, and attach ancillary
   rasters (water occurrence, land cover, DEM, HAND, optional GFM
   exclusion). Ancillary tiles are pulled from public cloud sources by
   default and cached on disk. Pass --ancillary-download-folder DIR to
   use a local cache, or set SIENA_OFFLINE=1 with pre-downloaded tiles.

2. Pixel-level detection — Pixel_level_process
   Score every pixel as water-like (fuzzy likelihood), then cut three
   nested masks. Thresholds are in Pixel_level_process.thSeg4 (not CLI)::

     high-probability water      normalized score > 0.80
     moderate-probability water  normalized score > 0.63
     low-probability water       normalized score > 0.51

3. Object-level refinement — Object_level_process
   Group pixels into water bodies and include or exclude each body with
   rule-based ratios (urban fraction, HAND, slope, desert/snow, high vs
   moderate overlap, bimodality). Those ratios are CLI flags below and
   are summarized in README.md.

------------------------------------------------------------------------
Parameters users often tune
------------------------------------------------------------------------

Pixel-level (edit Pixel_level_process.py, function thSeg4):
  0.80 / 0.63 / 0.51  high / moderate / low probability water cuts.

Object-level (CLI; see parse() and README.md):
  --rNWB_WST_Org                 urban (built-up) fraction to drop a body
  --object_hand_max_m            median HAND (m) above which a body is dropped
  --object_slope_max_deg         median slope (deg) above which a body is dropped
  --object_desert_*_ratio        bare/snow fraction to drop high/mod/low bodies
  --object_high_mask_preserve_ratio
  --object_remove_high_max / --object_remove_mod_max
  --object_moderate_clean_high_max / --object_moderate_clean_mod_max
  --permanent_water_wo_threshold
  --object_pwhw_high_min_ratio

Ancillary:
  --ancillary-download-folder DIR   local cache of GSW/LCC/DEM/HAND/GFM tiles
  --skip_gfm_exclusion              do not fetch the GFM exclusion mask
"""
import json
import os
import sys
import argparse
import numpy as np
import rasterio
from utils import tim, Timer, reproject_clip_direct
from Load_Data import Load_Data
from Pixel_level_process import run_pixel_level_process
from Object_level_process import run_object_level_process
from quick_look import quick_look

import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.json,.geojson")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "1048576")


def mosaic_pre_sim(output):
    """
    Build a pre-event / permanent-water reference on the flood scene.

    Single-image runs use water-occurrence pixels above
    permanent_water_wo_threshold. Multi-image runs mosaic moderate-probability
    water from the extra (dry) scenes onto the first (flood) scene grid.
    """
    if len(output) == 1 and getattr(output[0][0], "MODE", None) == "normal":
        output[0][0].floodimage = False
        print("No dry images, consider changing the mode to single!")
    elif len(output) == 1 or getattr(output[0][0], "MODE", None) == "single":
        wo_th = getattr(output[0][0], "permanent_water_wo_threshold", 75)
        output[0][1].pre_mod_mask = np.where(np.asarray(output[0][1].water_occurrence) > wo_th, 1, 0).astype(np.uint8)
    else:
        flood_src = output[0][0].ref_src
        dry_srcs = [out[0].ref_src for out in output[1:]]
        mod_masks = [out[1].Mod_mask for out in output[1:]]
        mosaic_results = []
        for dry_ras, dry_src in zip(mod_masks, dry_srcs):
            dry_src_meta = dry_src.meta.copy()
            dry_src_meta.update(count=1, dtype=rasterio.uint8, nodata=255)
            arr = reproject_clip_direct(np.asarray(dry_ras), dry_src_meta, flood_src.meta.copy())
            mosaic_results.append(arr)
        stacked = np.stack(mosaic_results, axis=0)
        suma = np.sum(stacked, axis=0)
        if suma.ndim == 3:
            suma = suma.squeeze(axis=0)
        output[0][1].pre_mod_mask = np.where(suma > 0, 1, 0).astype(np.uint8)
    return output


def _strip_ancillary_download_cli(argv_list):
    """
    Remove --ancillary-download-folder from argv so positional modes (tif + OUT_PATH) work
    even when the flag appears after paths. Returns (cleaned_argv, folder_or_None).
    """
    ancillary = None
    for i, a in enumerate(argv_list):
        if a == "--ancillary-download-folder" and i + 1 < len(argv_list):
            ancillary = argv_list[i + 1]
            break
        if a.startswith("--ancillary-download-folder="):
            ancillary = a.split("=", 1)[1]
            break
    out = []
    i = 0
    while i < len(argv_list):
        if argv_list[i] == "--ancillary-download-folder":
            i += 2
            continue
        if argv_list[i].startswith("--ancillary-download-folder="):
            i += 1
            continue
        out.append(argv_list[i])
        i += 1
    return out, ancillary


def parse(config_json=None):
    """
    Parse CLI flags and an optional JSON config (filepath_list / OUT_PATH).

    Object-level include/exclude ratios live here. Pixel-level high / moderate /
    low probability cuts (0.80 / 0.63 / 0.51) are in Pixel_level_process.thSeg4.
    """
    parser = argparse.ArgumentParser(
        description=(
            "SIENA baseline SAR flood mapping. "
            "PI: Qing Yang (henryqy@umd.edu). "
            "Citation: Yang Q (2026) Front. Water 8:1871753. doi: 10.3389/frwa.2026.1871753"
        )
    )
    parser.add_argument("--MODE", type=str, default="single", help="Run mode: single (default).")
    parser.add_argument("--morpho_seed_type", type=str, default="both", help="Reserved; not used by the current object-level path.")
    # ESA World Cover: 50 = Built-up (impervious), 80 = Permanent water bodies.
    parser.add_argument("--nWBClasses", default=[50], help="ESA WorldCover classes treated as non-water (default 50 = built-up). Used as the urban fraction in object rules.")
    parser.add_argument("--WBClasses", default=[80], help="ESA WorldCover water class (80 = permanent water bodies). Informational; PWA uses --pwa_lcc.")
    parser.add_argument("--rNWB_WST_Org", type=float, default=0.3, help="Object rule: drop a low-mask body if built-up fraction >= this (default 0.3 = 30%%).")
    parser.add_argument("--rNWB_WST", type=float, default=0.4, help="Reserved; not used by the current object-level path.")
    parser.add_argument("--rNWB_ICD", type=float, default=0.8, help="Reserved; not used by the current object-level path.")
    parser.add_argument("--thSize", type=int, default=10, help="Reserved; not used by the current object-level path.")
    parser.add_argument("--thInund", type=float, default=0.3, help="Reserved; not used by the current object-level path.")
    parser.add_argument("--rInund", type=float, default=1, help="Reserved; not used by the current object-level path.")
    parser.add_argument("--rProb", type=float, default=0.5, help="Reserved; not used by the current object-level path.")
    parser.add_argument("--floodimage", default=False, help="Internal: first granule is treated as the flood scene.")
    parser.add_argument("--skip_gfm_exclusion", default=True, action="store_true", help="Do not fetch the GFM SAR exclusion mask (default: skip).")
    parser.add_argument("--brd_erosion_pixels", type=int, default=0, help="Erode the valid-data mask by N pixels to drop noisy image edges (0 = off).")
    parser.add_argument("--use_incid_normalization", default=False, action="store_true", help="NetCDF: flatten range trend as sigma / cos(incidence)^k when incid is present (default: off).")
    parser.add_argument("--nc_mask_water_as_nodata", default=False, action="store_true", help="NetCDF: treat producer mask water (negative) as no-data so only land is mapped (default: off).")
    parser.add_argument("--use_region_growth", default=False, action="store_true", help="Grow water from high-probability seeds into the low-score neighborhood instead of high/low object intersection (default: intersection).")
    parser.add_argument("--object_hand_max_m", type=float, default=20.0, help="Object rule: drop a low-mask body if median HAND (m) > this (default 20).")
    parser.add_argument("--object_slope_max_deg", type=float, default=20.0, help="Object rule: drop a low-mask body if median slope (deg) > this (default 20).")
    parser.add_argument("--lcc_bare", type=int, default=60, help="ESA WorldCover class for bare / sparse vegetation in desert/snow removal (default 60).")
    parser.add_argument("--object_desert_high_ratio", type=float, default=0.6, help="Drop a high-probability body if bare/snow fraction >= this (default 0.6 = 60%%).")
    parser.add_argument("--object_desert_mod_ratio", type=float, default=0.5, help="Drop a moderate-probability body if bare/snow fraction >= this (default 0.5 = 50%%).")
    parser.add_argument("--object_desert_low_ratio", type=float, default=0.4, help="Drop a low-probability body if bare/snow fraction >= this (default 0.4 = 40%%).")
    parser.add_argument("--object_wo_threshold", type=float, default=5.0, help="Optional nearby-water-occurrence rule (implemented but not on the default path): WO > this counts as water history.")
    parser.add_argument("--object_wo_min_ratio", type=float, default=0.01, help="Optional nearby-water-occurrence rule (not on the default path): required WO fraction in a buffer around the object.")
    parser.add_argument("--object_high_mask_preserve_ratio", type=float, default=0.35, help="Keep a flood-water body in full if its high-probability pixel rate >= this (default 0.35).")
    parser.add_argument("--object_remove_high_max", type=float, default=0.03, help="Drop a flood-water body if high-probability rate < this OR moderate rate < --object_remove_mod_max (default 0.03).")
    parser.add_argument("--object_remove_mod_max", type=float, default=0.15, help="Drop a flood-water body if moderate-probability rate < this OR high rate < --object_remove_high_max (default 0.15).")
    parser.add_argument("--object_moderate_clean_high_max", type=float, default=0.10, help="After bimodality: if high rate < this AND moderate rate < --object_moderate_clean_mod_max, keep only the moderate-mask extent (default 0.10).")
    parser.add_argument("--object_moderate_clean_mod_max", type=float, default=0.25, help="After bimodality: moderate-rate cut used with --object_moderate_clean_high_max (default 0.25).")
    parser.add_argument("--permanent_water_wo_threshold", type=float, default=75.0, help="Water occurrence (0-100) above this is treated as permanent / pre-event water in the display product (default 75).")
    parser.add_argument("--pwa_lcc", nargs="*", type=int, default=[80, 90], help="ESA WorldCover classes that define permanent-water area with high occurrence (default: 80 90).")
    parser.add_argument("--object_pwhw_high_min_ratio", type=float, default=0.30, help="Herbaceous wetland (ESA 90): keep a low-mask body only if high-probability overlap >= this (default 0.30).")
    parser.add_argument("--output_intermediate", action="store_true", default=False, help="Write intermediate/debug rasters (default: only SIENA_2classes_RGB, SIENA_raw, and the JPEG).")
    parser.add_argument(
        "--ancillary-download-folder",
        type=str,
        default=None,
        metavar="DIR",
        help="Local ancillary cache (GSW_download, LCC_download, DEM_download, HAND30_download, GFM_exclusion_download). Default: <code>/ancillary_download or ANCILLARY_ROOT/ancillary_download. Tiles are downloaded from public cloud URLs when missing.",
    )
    args, _ = parser.parse_known_args()
    if getattr(args, "ancillary_download_folder", None):
        import ancillarydata_merge as _anc

        _anc.set_ancillary_download_folder(args.ancillary_download_folder)
    # If a JSON config is provided, use it for filepath_list and optional OUT_PATH.
    if config_json is not None and os.path.exists(config_json):
        with open(config_json) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            if "filepath_list" in loaded:
                args.filepath_list = loaded["filepath_list"]
            elif "filepath" in loaded:
                # Allow simple configs with a single filepath or list of filepaths
                fp = loaded["filepath"]
                if isinstance(fp, str):
                    args.filepath_list = [[fp]]
                elif isinstance(fp, list) and fp and isinstance(fp[0], str):
                    args.filepath_list = [fp]
            # Output path: support OUT_PATH, output_path, or out_path
            if "OUT_PATH" in loaded and loaded["OUT_PATH"]:
                args.OUT_PATH = os.path.abspath(os.path.expanduser(loaded["OUT_PATH"]))
            elif "output_path" in loaded and loaded["output_path"]:
                args.OUT_PATH = os.path.abspath(os.path.expanduser(loaded["output_path"]))
            elif "out_path" in loaded and loaded["out_path"]:
                args.OUT_PATH = os.path.abspath(os.path.expanduser(loaded["out_path"]))
    return args


def dup_args(args, filepath, granuele_name):
    """Copy run settings for one input file and set that granule's output folder."""
    args_new = argparse.Namespace(**vars(args))
    args_new.filepath = filepath
    args_new.dirOut = os.path.join(args_new.WORK_PATH, granuele_name)
    args_new.fileSampleMap = os.path.join(args_new.dirOut, "SampleMap.tif")
    args_new.logFile = os.path.join(args_new.dirOut, "sta_log.csv")
    # dirOut is created in run_siena loop (flat for single granule, subfolder for multi)
    args_new.chunks = getattr(args, "blockSize", 1024)
    return args_new


@tim
def run_siena(*args_main, **kwargs):
    """
    Run the three-stage SIENA baseline: initialization, pixel-level detection,
    object-level refinement, then a quick-look map.

    Usage:
      SIENA.py <input.nc|input.tif> <OUT_PATH>
      SIENA.py <co_pol.tif> <cross_pol.tif> <OUT_PATH>
      SIENA.py <config.json>
      SIENA.py <input.tif>   # output defaults to <dir>/siena_out/
      Optional: --ancillary-download-folder DIR  (any position)
    """
    argv_list, anc_from_pos = _strip_ancillary_download_cli(list(args_main))
    if anc_from_pos:
        import ancillarydata_merge as _anc

        _anc.set_ancillary_download_folder(anc_from_pos)
    args_main = tuple(argv_list)

    if len(args_main) == 1:
        one = args_main[0]
        if one.lower().endswith(".tif") or one.lower().endswith(".tiff") or one.lower().endswith(".nc"):
            # Single co-pol GeoTIFF or NetCDF → ./siena_out/<basename>
            if not os.path.exists(one):
                raise ValueError("Input file does not exist: %s" % one)
            filepath = [one]
            OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(one)), "siena_out")
            args = parse(None)
            args.OUT_PATH = OUT_PATH
            args.filepath_list = [filepath]
        else:
            config_file = one
            if not os.path.exists(config_file):
                raise ValueError("Config file does not exist: %s" % config_file)
            args = parse(config_file)
            if not getattr(args, "filepath_list", None):
                raise ValueError("Config must contain filepath_list or filepath")
            # Default OUT_PATH from JSON or first input file's directory
            if not getattr(args, "OUT_PATH", None):
                first = args.filepath_list[0][0]
                args.OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(first)), "siena_out")
                print("OUT_PATH not in config; using default: %s" % args.OUT_PATH)
    elif len(args_main) == 2:
        # Single co-pol: <co_pol.tif> <OUT_PATH>
        filepath = [args_main[0]]
        OUT_PATH = args_main[1]
        args = parse(None)
        args.OUT_PATH = OUT_PATH
        args.filepath_list = [filepath]
    elif len(args_main) == 3:
        # Dual-pol: <co_pol.tif> <cross_pol.tif> <OUT_PATH>
        filepath = [args_main[0], args_main[1]]
        OUT_PATH = args_main[2]
        args = parse(None)
        args.OUT_PATH = OUT_PATH
        args.filepath_list = [filepath]
    else:
        raise ValueError(
            "Usage: SIENA.py <config.json>  |  SIENA.py <input.nc|input.tif> <OUT_PATH>  |  "
            "SIENA.py <co_pol.tif> <cross_pol.tif> <OUT_PATH>"
        )

    # Output under OUT_PATH / image_name (one folder per granule)
    args.WORK_PATH = args.OUT_PATH
    filepath_list = args.filepath_list
    args_list = []
    for idx, filepath in enumerate(filepath_list):
        granuele_name = os.path.splitext(os.path.basename(filepath[0]))[0]
        args1 = dup_args(args, filepath, granuele_name)
        args1.granules = granuele_name
        args1.dirOut = os.path.join(args.WORK_PATH, granuele_name)
        os.makedirs(args1.dirOut, exist_ok=True)
        args_list.append(args1)
    args_list[0].floodimage = True

    if getattr(args_list[0], "MODE", "single") == "desert":
        print("Desert mode is not implemented in this public baseline.")
        return

    output = []

    # -------------------------------------------------------------------------
    # 1. Initialization: load SAR + ancillary (cloud cache or local folder)
    # -------------------------------------------------------------------------
    print("[Initialization]")
    with Timer("Load_Data Total"):
        for arggs in args_list:
            args_out, ds = Load_Data(arggs)
            output.append((args_out, ds))

    # -------------------------------------------------------------------------
    # 2. Pixel-level detection: fuzzy score → high / moderate / low probability masks
    # -------------------------------------------------------------------------
    print("[Pixel_level_process]")
    with Timer("Pixel_level_process"):
        output = [run_pixel_level_process(o[0], o[1]) for o in output]
        output = [(o[0], o[1]) for o in output]

    # -------------------------------------------------------------------------
    # 3. Object-level refinement: rule-based include/exclude of water bodies
    # -------------------------------------------------------------------------
    print("[Object_level_process]")
    output = mosaic_pre_sim(output)
    with Timer("Object_level_process"):
        output = [run_object_level_process(o[0], o[1]) for o in output]
        output = [(o[0], o[1]) for o in output]

    out_filepath = quick_look(output)
    return out_filepath


if __name__ == "__main__":
    run_siena(*sys.argv[1:])
