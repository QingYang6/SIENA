# SIENA

This repository is a **baseline SAR flood-mapping** workflow for public trial.

**SIENA** (Scientific Inundation Evolution Network Agent for SAR) is an AI architect. It built this baseline so that a general user can map surface water and flood inundation from a single SAR image. The intended starting point is simple: provide a SAR file, run one command, and inspect the flood map.

**PI:** Qing(Henry) Yang (`henryqy@umd.edu`)

**Paper (please cite):**

Yang Q (2026) Operational SAR flood mapping as a full-stack systems problem: an AI-enabled perspective. *Front. Water* 8:1871753. doi: [10.3389/frwa.2026.1871753](https://doi.org/10.3389/frwa.2026.1871753)

---

## How to use

### 1. Install

Use conda-forge (includes GDAL and Python bindings):

```bash
conda install -c conda-forge gdal rasterio scipy requests shapely pyproj psutil pandas netcdf4 opencv matplotlib pystac-client
```

### 2. Check the environment

```bash
python check_env.py
```

This checks Python imports, creates ancillary cache folders, and tests whether public ancillary sources are reachable. On success you should see: `Environment check passed.`

### 3. Run

Single co-pol input. **GeoTIFF or NetCDF (.nc)** are both accepted:

```bash
python SIENA.py input.nc outputfolder
```

or

```bash
python SIENA.py input.tif outputfolder
```

Optional dual-pol (co-pol then cross-pol):

```bash
python SIENA.py co_pol.tif cross_pol.tif outputfolder
```

Optional: reuse a local ancillary cache (folder that already contains `GSW_download`, `LCC_download`, and the other `*_download` subfolders):

```bash
python SIENA.py --ancillary-download-folder /path/to/ancillary_download input.tif outputfolder
```

See `python SIENA.py --help` for object-level rule ratios and other flags. Pixel-level water-probability thresholds are documented below and live in `Pixel_level_process.py`.

### Output

Results are written under a subfolder named after the input file:

```
<outputfolder>/
└── <inputfile_name>/
    ├── SIENA_2classes_RGB_<inputfile_name>.tif
    ├── SIENA_raw_<inputfile_name>.tif
    └── SIENA_2classes_<inputfile_name>.jpg
```

- `SIENA_raw_*.tif` — binary water / non-water mask
- `SIENA_2classes_RGB_*.tif` / `.jpg` — map with permanent/pre-event water vs flood inundation over land cover

The first run for a new region downloads ancillary tiles into `ancillary_download/` (or `ANCILLARY_ROOT` / `--ancillary-download-folder`). Later runs over the same area reuse that cache.

---

## Input

The loader is meant to be flexible:

| Input | What SIENA uses |
|--------|------------------|
| **GeoTIFF** (`.tif` / `.tiff`) | First band as co-pol backscatter. A second band, or a second file, can be cross-pol. |
| **NetCDF** (`.nc`) | Variable `sigma` (linear NRCS), plus `latitude` / `longitude` when present. See [README_NRCS_NC.md](README_NRCS_NC.md) for the expected fields. |
| **HTTPS zip URL** | A zip of RTC-style GeoTIFFs (VV/HH and optional VH/HV). |

Single-pol scenes are duplicated internally so the rest of the pipeline can run unchanged.

---

## Pipeline at a glance

The entry point is `SIENA.py`. It always runs three stages:

1. **Initialization** (`Load_Data.py`) — read the SAR image, place it on a regular geographic grid if needed, and attach ancillary layers.
2. **Pixel-level detection** (`Pixel_level_process.py`) — score every pixel as water-like, then cut three nested probability masks (high / moderate / low).
3. **Object-level refinement** (`Object_level_process.py`) — group pixels into water bodies and include or exclude each body with hydrologic and land-cover rules.

A short quick-look step then writes the GeoTIFF and JPEG products.

The same three-stage description is in the header of `SIENA.py`. Tunable parameters are listed there and in the sections below.

---

## Ancillary data (online or local)

SIENA does not ship global ancillary rasters. For each scene it clips and reprojects:

| Layer | Source (default online) | Used for |
|--------|-------------------------|----------|
| Water occurrence | Global Surface Water (occurrence tiles) | Permanent / pre-event water, fuzzy score |
| Land cover | ESA WorldCover 2020 | Urban, bare, snow, wetland, permanent water classes |
| DEM | Copernicus GLO-30 | Slope |
| HAND | GLO-30 HAND | Height above nearest drainage |
| GFM exclusion (optional) | EODC STAC (Global Flood Monitoring) | Drop pixels that the GFM exclusion mask flags as unreliable SAR water |

**Online (default).** Tiles are downloaded from public cloud URLs and cached under `ancillary_download/` next to the code, or under `ANCILLARY_ROOT` if that environment variable is set. You can also pass `--ancillary-download-folder DIR`.

**Offline.** If tiles are already in those folders, SIENA reuses them. On a machine with no internet, set `SIENA_OFFLINE=1` and pre-stage the same folder layout (run `check_env.py` and one online scene first, or copy a cache from another machine).

GFM exclusion can be skipped with `--skip_gfm_exclusion` (this is the default in the current CLI).

---

## Parameters you may want to tune

Two groups matter most. Both are also summarized in `SIENA.py`.

### Pixel-level: three water-probability thresholds

After the fuzzy water score is normalized to `[0, 1]`, `thSeg4` in `Pixel_level_process.py` builds three nested masks. These are **not** CLI flags today; change the three numbers in that function if you want a more aggressive or more conservative detector.

| Mask | Meaning | Default cut on normalized score |
|------|---------|----------------------------------|
| High-probability water | Strongest water-like pixels (seeds) | `> 0.80` |
| Moderate-probability water | Intermediate water-like pixels | `> 0.63` |
| Low-probability water | Broadest candidate water | `> 0.51` |

Because the cuts are nested (`0.80 > 0.63 > 0.51`), high ⊂ moderate ⊂ low. Raising the high cut makes fewer seed objects; lowering the low cut grows more candidate water that object rules must later accept or reject.

Related (also in code, not CLI): tile-based initial thresholding in `initial_threshold_core.py` (Otsu on SAR intensity in tiles that mix water and land), and fuzzy membership ranges in `Fuzzy_logic_core.get_default_fuzzy_option()` (HAND, slope, water occurrence).

### Object-level: rule-based include / exclude

Connected water bodies are kept or dropped by ratios and ancillary tests. These **are** CLI flags on `SIENA.py`. The idea is: a body that looks like water in SAR may still be urban shadow, steep terrain, bare desert, or a weak speckle blob — the rules decide.

| Flag | Default | What it does |
|------|---------|----------------|
| `--rNWB_WST_Org` | `0.3` | Drop a low-mask object if its **built-up (urban) fraction** ≥ this (30%). |
| `--object_hand_max_m` | `20` | Drop a low-mask object if **median HAND** (m) is above this. |
| `--object_slope_max_deg` | `20` | Drop a low-mask object if **median slope** (degrees) is above this. |
| `--object_desert_high_ratio` | `0.6` | Drop a **high**-mask object if bare/snow fraction ≥ this. |
| `--object_desert_mod_ratio` | `0.5` | Same test on the **moderate** mask. |
| `--object_desert_low_ratio` | `0.4` | Same test on the **low** mask (after other cleaning). |
| `--object_high_mask_preserve_ratio` | `0.35` | Keep a flood-water object in full if its high-probability pixel rate ≥ this. |
| `--object_remove_high_max` | `0.03` | Drop a flood-water object if high-probability rate is below this **or** the moderate rate is below `--object_remove_mod_max`. |
| `--object_remove_mod_max` | `0.15` | See previous row. |
| `--object_moderate_clean_high_max` | `0.10` | After a bimodality check: if high rate and moderate rate are both below their cuts, keep only the moderate-mask extent of that object. |
| `--object_moderate_clean_mod_max` | `0.25` | See previous row. |
| `--permanent_water_wo_threshold` | `75` | Pixels with water occurrence above this (0–100), or ESA class 80, are treated as permanent / pre-event water in the display product. |
| `--object_pwhw_high_min_ratio` | `0.30` | In herbaceous wetland (ESA 90), keep a low object only if high-probability overlap ≥ this. |
| `--nWBClasses` | `[50]` | ESA WorldCover classes treated as non-water (default 50 = built-up). |
| `--pwa_lcc` | `[80, 90]` | ESA classes that define the permanent-water area together with high water occurrence. |

Default path: keep **low**-probability objects that **intersect** a **high**-probability object, then apply the preserve / remove / bimodality rules. Optional `--use_region_growth` grows from high seeds into the low-score neighborhood instead of that intersection.

NetCDF-only options: `--use_incid_normalization`, `--nc_mask_water_as_nodata`, `--brd_erosion_pixels`. Debug rasters: `--output_intermediate`.

---

## Technical notes (read this if you are changing the code)

### Initialization (`Load_Data.py`)

- Opens GeoTIFF, NetCDF, or an HTTPS zip.
- For `.nc`, reads `sigma` and forward-maps it onto a regular WGS84 grid from per-pixel lat/lon (see [README_NRCS_NC.md](README_NRCS_NC.md)).
- Calls `get_wo`, `get_lcc`, `get_dem`, `get_HAND30`, and optionally `get_gfm_exclusion` in `ancillarydata_merge.py`. Those functions look up intersecting tiles, download if missing, and reproject onto the SAR grid.

### Pixel-level detection (`Pixel_level_process.py`)

1. `initial_sample` — sets simple intensity bounds (`I1u`, `I2u`) from percentiles of valid backscatter.
2. `run_fuzzy_core` — tile Otsu initialization, then a fuzzy score from dark SAR + HAND + slope + water occurrence.
3. `thSeg4` — normalize the fuzzy score and apply the three probability cuts; drop tiny speckle groups on the high mask.

Supporting files: `Initial_sample.py`, `Fuzzy_logic_core.py`, `initial_threshold_core.py`.

### Object-level refinement (`Object_level_process.py`)

`run_object_level_process` is the rule engine:

1. Split candidates into **flood-water (FW)** vs **permanent-water area (PWA)**.
2. On FW only: urban / HAND / slope filters on low-mask objects; desert/snow filters on high, moderate, and low masks.
3. Keep low-mask FW objects that overlap high-mask FW objects.
4. Inside PWA: permanent water (ESA 80 + high occurrence) uses high–low intersection; herbaceous wetland (ESA 90) requires a minimum high-mask fraction.
5. Classify remaining FW objects as preserve, remove, or check; run a SAR-intensity **bimodality** test on the check set.
6. Optionally apply the GFM exclusion mask; write `SIENA_raw_*.tif`.

`quick_look.py` then colors permanent water vs flood for the RGB / JPEG product.

### Other modules

| File | Role |
|------|------|
| `ancillarydata_merge.py` | Cloud or local ancillary tile lists, download, merge onto the SAR grid |
| `gfm_sar_exclusion_mars.py` | GFM exclusion mask from EODC STAC (MARS fetch is optional and unused by default) |
| `utils.py` | Raster I/O, reprojection, timers |
| `check_env.py` | Import and ancillary-reachability check |

Function-level comments in each file describe the local step, not a second copy of this README.

---

## Citation

```
Yang Q (2026) Operational SAR flood mapping as a full-stack systems problem:
an AI-enabled perspective. Front. Water 8:1871753.
doi: 10.3389/frwa.2026.1871753
```

Paper: https://doi.org/10.3389/frwa.2026.1871753

Contact: Qing(Henry) Yang, `henryqy@umd.edu`
