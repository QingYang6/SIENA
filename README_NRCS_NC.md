# Expected structure of NetCDF (.nc) SAR input

SIENA accepts geocoded Level-2 NetCDF files (for example NOAA/NESDIS NRCS
products such as SAROPS) as well as GeoTIFF. This note lists the fields the
loader looks for so it can use them when present and fall back when they are
missing. For how to run SIENA, start with [README.md](README.md).

## Source / convention

- **Typical source**: SAR Operational Product System (SAROPS), e.g. NOAA/NESDIS NRCS Level-2 products (RADARSAT-2, RCM, S1, etc.).
- **Conventions**: CF-1.6.
- **Content**: Calibrated Normalized Radar Cross Section (sigma0), geocoded to WGS84 with per-pixel lat/lon and optional incidence/look.

---

## Dimensions

| Dimension | Meaning | Use in SIENA |
|-----------|---------|--------------|
| `x` | Range (ground) direction, image columns | Width of sigma/lat/lon |
| `y` | Azimuth direction, image rows | Height of sigma/lat/lon |
| `xlon`, `xlat`, `xi`, `xj`, `xincid`, `xrlook` | Polynomial coefficient sizes (e.g. 6) | Not read; used by producer for geo/incid |

---

## Required / main variables for SIENA

| Variable | Shape | Type | Meaning | Use in SIENA |
|----------|--------|------|---------|---------------|
| **`sigma`** | (y, x) | float32 | Normalized Radar Cross Section (linear, units=1) | **Primary backscatter** → `img_lp`; duplicated to `img_cp` for single-pol. **Already linear**; no dB→linear conversion. |
| **`latitude`** | (y, x) | float32 | Latitude (degrees) per pixel | Bounds (south/north), georeference, and **left–right orientation** (flip if col 0 is east of last col). |
| **`longitude`** | (y, x) | float32 | Longitude (degrees) per pixel | Bounds (west/east), georeference, and orientation check. |

If `latitude` and `longitude` are missing, global corner attributes can be used (see below).

---

## Optional variables (used when present and valid)

| Variable | Shape | Type | Meaning | Use in SIENA |
|----------|--------|------|---------|---------------|
| **`incid`** | (y, x) | float32 | Incidence angle (degrees) per pixel | Future: incidence-based normalization or masking; currently stored as `ds.inc` if needed. |
| **`mask`** | (y, x) | int16 | Land/water (positive=land, negative=water) | Could be used for valid/data mask; if missing, use sigma > 0 and erosion. |
| **`variance`** | (y, x) | float32 | Variance of NRCS | Speckle/ENL; not yet used in SIENA. |

---

## Global attributes (for provenance and fallbacks)

| Attribute | Meaning | Fallback in SIENA |
|-----------|---------|-------------------|
| `ul_outer_lon`, `ur_outer_lon`, `ll_outer_lon`, `lr_outer_lon` | Corner longitudes | If `longitude` missing: bounds and orientation (flip if ul_lon > ur_lon). |
| `ul_outer_lat`, `ur_outer_lat`, `ll_outer_lat`, `lr_outer_lat` | Corner latitudes | If `latitude` missing: bounds. |
| `polarization` | e.g. "VV", "HH" | Single-pol → duplicate to img_cp. |
| `platform_name` | e.g. "RADARSAT-2", "RCM-1" | Logging only. |
| `time_coverage_start`, `time_coverage_end` | Acquisition time (UTC) | Optional for GFM or time-based logic. |
| `process_level` | e.g. "Level 2" | Expect calibrated, geocoded product. |

---

## Scalar / metadata variables (not required for SIENA load)

- **Calibration**: `scaling` (NRCS / DN).
- **Resolution**: `nominal_azimuth_resolution_m`, `nominal_range_resolution_m`, `pixel_spacing_m`, `line_spacing_m`.
- **Geometry**: `heading_north_deg`, `incid_angle_center_deg`, `slant_range_center_m`, etc.
- **Polynomials**: `lon_coef`, `lat_coef`, `incid_coef`, etc. (for producer’s geo model).

These are part of the standard NRCS .nc but SIENA does not read them for the initial load; resolution is derived from the geocoded grid (bounds and dimensions) when needed.

---

## Behavior in SIENA when input is .nc

1. **Load**: Read `sigma(y, x)` and, when present, 2D `latitude`/`longitude`.
2. **Georeference (when 2D lat/lon exist)**  
   **GAMMA-style forward mapping (default):** The target is a regular WGS84 affine grid (same bounds and dimensions as the NC, from `from_bounds`). Each **NC pixel** (i, j) with position (lon[i,j], lat[i,j]) and value sigma[i,j] **contributes** to the target: we compute which target pixel (row, col) that position falls in (`rowcol(transform, lon, lat)`), then accumulate sigma there. When several NC pixels land in the same target pixel we **average** them. So the value at each target pixel comes only from NC pixels whose location falls in that pixel—no inverse interpolation, no fill outside the footprint. Pixels with no contribution stay 0. This matches the idea of “resample onto a standard affine using the per-pixel geo from the source.”
3. **Georeference (fallback when only corner attrs)**  
   Use bounds from corner attributes; correct flip_x/flip_y; build transform with `from_bounds`. May misalign if the NC grid is non-affine.
4. **NC metadata for a better affine (future):** If the .nc provides `pixel_spacing_m`, `line_spacing_m`, and a reference corner (e.g. first-pixel lat/lon) or polynomial coeffs (`lon_coef`, `lat_coef`), a target grid could be built from that instead of `from_bounds` to reduce residual. Not implemented yet; current exact resample + footprint mask is the default.
5. **Export**: Write `sar_from_nc.tif` (linear) and `sar_from_nc_dB.tif` (dB, nodata -50). Same grid for `ref_src` and ancillary (WO, LCC, DEM, HAND, GFM).
6. **Values**: Use sigma **as-is (linear)**; do **not** apply dB→linear. Set `ds.from_nc = True` so downstream skips range-profile normalization.
7. **Single-pol**: If only one polarization, set `img_cp = img_lp.copy()`.
8. **Missing/invalid**: If `latitude`/`longitude` missing, use corner attributes; if those missing, use safe defaults.

---

## Option B: Warp ancillary to the NC grid (design)

To remove residual mismatch and edge stretch entirely, **keep sigma on the NC native (ny, nx) grid** and **sample ancillary at each (lon[i,j], lat[i,j])** so every matrix operation uses the same pixel grid.

- **Reference grid:** NC shape (ny, nx) and 2D arrays `lat`, `lon` (no single affine).
- **Ancillary:** For WO, LCC, DEM, HAND, GFM: instead of `reproject_and_merge(..., input_data_meta)`, call a path that **samples** the merged raster at `(lat_2d, lon_2d)`. Helper in `utils.py`: `sample_raster_at_latlon(raster_data_dict, lat_2d, lon_2d)` returns an array of shape (ny, nx) with values at those geographic points. So each ancillary would: (1) fetch/merge tiles as today into a raster covering the NC bounds, (2) sample that raster at (lat_2d, lon_2d) → array (ny, nx).
- **Load_Data change:** When input is .nc and Option B is enabled, do **not** call `_geocode_nc_exact_resample`; keep `sigma`, `lat_2d`, `lon_2d` as-is. Build `SAR_img_d_m` with a placeholder meta (e.g. from_bounds for bounds/crs only) and pass `lat_2d`/`lon_2d` (e.g. on `ds`) so ancillary layer can branch: if “NC native grid” then sample at points instead of reproject.
- **Output GeoTIFFs/figures:** All arrays are (ny, nx). To write a GeoTIFF we need a geotransform: use a **from_bounds** transform (approximate) so the image displays in the right place in QGIS; pixel locations are then “close” but not a perfect standard grid. Alternatively write with GCPs (subsampled) for better display.

This requires branching the ancillary entry points (get_wo, get_lcc, etc.) and ensuring every consumer of `ds` (binary classification, fuzzy, morpho, compensate, quick_look) only uses the arrays, not the reference transform for pixel-level logic. The helper `sample_raster_at_latlon` is implemented in `utils_merge` for use when Option B is wired in.

---

## Alignment with GAMMA (we do the same thing)

GAMMA (and similar SAR processors) geocode by:

1. **Computing** the mapping (line, pixel) → (lat, lon) from the sensor model (and DEM for terrain).
2. **Choosing a target grid** that is a **standard affine** (e.g. DEM at 90 m or 100 m, or a fixed WGS84 resolution).
3. **Resampling** the SAR onto that target: each **source** pixel contributes to the target pixel its (lat, lon) falls in (forward mapping).

We do **the same resampling step**: target = regular WGS84 affine; each NC pixel (lat[i,j], lon[i,j]) contributes sigma[i,j] to the target pixel from `rowcol(transform, lon, lat)`; multiple contributions are averaged. So we are **not** reinventing the wheel—same “target affine + forward revalue from source geo.”

**Only difference:** We do **not** have a standard DEM or product grid at 90 m / 100 m to match. So we **define a new affine at close resolution** from the NC (same bounds and dimensions → resolution ≈ span/nx, span/ny). The SAR is revalued onto that grid exactly like GAMMA. Because this grid is **not** a standard DEM or 90 m/100 m product grid, there can be a **natural slight mismatch** with ancillary data (which are on their own tile grids and resolutions). The SAR itself is revalued correctly; the remaining uncertainty is mainly grid/resolution alignment with ancillary.

---

## Uncertainty sources (things to keep an eye on)

| Source | Description | Mitigation / note |
|--------|-------------|-------------------|
| **Target grid not standard** | Our affine is from NC bounds + (nx, ny), so resolution is “close” but not 90 m / 100 m DEM or a standard product grid. Ancillary (WO, LCC, DEM, HAND, GFM) are reprojected to this grid but come from different native resolutions and tile boundaries. | Expect sub-pixel to few-pixel level mismatch at boundaries; SAR is on a consistent grid, ancillary are sampled onto it. |
| **SAR geolocation in the NC** | (lat, lon) in the .nc come from the producer’s geo model. Any error there (orbit, DEM used by producer, timing) propagates into our target pixels. | Inherent to the product; no fix in SIENA. Check producer docs for stated geolocation accuracy. |
| **Multiple NC pixels → one target pixel** | When several NC pixels fall in the same target cell we **average** sigma. That can smooth edges or mix values. | Same as GAMMA when target is coarser than source; acceptable for flood mapping. |
| **Ancillary resolution and tiles** | WO (30 m tiles), LCC, DEM (30 m), HAND, GFM have their own resolutions and tile grids. Reprojecting them to our (non-standard) grid introduces resampling and possible edge effects. | Minor; keep an eye on coastlines or tile seams overlapping the scene. |
| **Single-pol and calibration** | Single-pol (e.g. VV only) duplicated to img_cp; thresholds and fuzzy logic are tuned for dual-pol. Calibration is as in the NC (we use sigma as-is). | Known limitation; use dual-pol when available. |
| **No terrain correction in our step** | We use the NC’s pre-geocoded (lat, lon). If the producer used a different DEM or no DEM, layover/shadow and local geometry can differ from our ancillary DEM. | Acceptable for many applications; for steep terrain consider producer’s DEM and product type. |

**Summary:** The main thing to watch is that the **SAR is revalued correctly** onto our affine (GAMMA-style); the remaining uncertainty is **grid/resolution alignment with ancillary** (non-standard target) and **product-level** geolocation and calibration. For flood mapping, these are usually acceptable; validate on a few scenes with known geography.

---

## Using extra NC fields (noise / normalization options)

When loading from .nc, SIENA can read optional variables and forward-map them to the same grid as sigma. These are then available on `ds` for downstream use or future options.

| Variable | On `ds` | Possible use |
|----------|---------|---------------|
| **`incid`** | `ds.inc` (when present) | **Incidence normalization:** flatten range trend with σ_corrected = σ / cos(inc)^k (e.g. k=0.5–1). Reduces range-dependent bias before fuzzy/classification. |
| **`mask`** | `ds.nc_mask` (when present) | **Valid-data mask:** use producer mask instead of (or combined with) sigma > 0 to exclude invalid/edge pixels and reduce noise. |
| **`variance`** | `ds.nc_variance` (when present) | **Speckle / ENL:** weight by 1/variance or mask low-ENL pixels; or use in adaptive filtering to reduce noise. |

**Implemented:**
- **A) Incidence normalization:** `--use_incid_normalization` (default off). When True and the .nc has `incid` with valid values (finite, 0–90°), apply σ ← σ / cos(inc_rad)^k (k=0.5) before downstream. **Skipped** if `incid` is missing, all nan, or all zero.
- **B) NC mask for valid pixels:** When the .nc has `mask`, it is geocoded to the same grid and used in `set_nan`: pixels where mask is 0 or non-finite are zeroed. **Skipped** if `mask` is missing or has no valid (finite, ≠0) pixels. Option `--nc_mask_water_as_nodata` (default: off) also treats water (negative mask) as no-data so only land is used for follow-up; disable by default because current mask quality may be limited.

**Guards:** Both steps use if/else so that if the data do not exist or are all no-data (zero/nan), the step is skipped and the pipeline continues unchanged.

---

## When input is not .nc (GeoTIFF / zip)

- Current behavior unchanged: rasterio open, optional dB→linear if filename contains `"nrcs"`, range-profile normalization in fuzzy logic, same ancillary and thresholds.
