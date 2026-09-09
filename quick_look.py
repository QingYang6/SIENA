"""
Quick-look products: binary water GeoTIFF coloring and a JPEG overview.

Permanent / pre-event water (high water occurrence or ESA class 80) is drawn
separately from flood inundation. Called at the end of SIENA.py.
"""
import os
import shutil
import numpy as np
import rasterio
from rasterio.transform import xy as transform_xy
from rasterio.warp import transform as warp_transform
from utils import reproject_clip_direct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from osgeo import gdal


def composition_sim(output):
    """If extra dry scenes were provided, mosaic their water masks as a pre-event reference."""
    if len(output) == 1 or getattr(output[0][0], "MODE", None) == "single":
        output[0][1].pre_compensate_mask = None
    else:
        flood_src = output[0][0].ref_src
        dry_srcs = [out[0].ref_src for out in output[1:]]
        mod_masks = [out[1].Compensate_mask for out in output[1:]]
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
        output[0][1].pre_compensate_mask = np.where(suma > 0, 1, 0).astype(np.uint8)
    return output


def purge2array_sim(output):
    """Pull flood mask, water occurrence, land cover, and valid-data arrays for the overview map."""
    flood_arr = np.asarray(output[0][1].Compensate_mask)
    WOP_arr = np.asarray(output[0][1].water_occurrence)
    LCC_arr = np.asarray(output[0][1].land_cover)
    valid_mask = output[0][1].img_lp > 0
    pre_comp = getattr(output[0][1], "pre_compensate_mask", None)
    dry_arr = np.asarray(pre_comp) if pre_comp is not None else None
    dire_arr = np.asarray(output[0][1].direct_mask) if getattr(output[0][1], "direct_mask", None) is not None else None
    return flood_arr, dry_arr, dire_arr, WOP_arr, LCC_arr, valid_mask


def quicklook_color():
    """ESA WorldCover palette plus product classes: pre-event water, flood inundation, optional ice."""
    # 11 ESA + 3 product classes; colors aligned with ESA-style land cover + flood/dry/ice
    palette_MergeProduct = np.array([
        [56, 168, 0],   # 10 Tree cover
        [163, 255, 115], # 20 Shrubland
        [76, 230, 0],   # 30 Grassland
        [230, 230, 0], # 40 Cropland
        [0, 0, 0],     # 50 Built-up
        [230, 152, 0], # 60 Bare / sparse vegetation
        [255, 190, 232], # 70 Snow and ice
        [0, 92, 230],  # 80 Permanent water bodies
        [0, 255, 197], # 90 Herbaceous wetland
        [0, 140, 140], # 95 Mangroves
        [178, 178, 178], # 100 Moss and lichen
        [0, 197, 255], # 110 Pre-event Water
        [219, 0, 0],   # 120 Flood inundation
        [255, 215, 0]  # 130 River Ice
    ])
    palette_MergeProductRGB = palette_MergeProduct / 255
    cmap = ListedColormap(palette_MergeProductRGB)
    # ESA 10,20,...,95,100 then product 110,120,130 (bin edges)
    norm = colors.BoundaryNorm([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 110, 120, 130, 140], 14)
    NameofLabel = [
        "Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up", "Bare / sparse vegetation",
        "Snow and ice", "Permanent water bodies", "Herbaceous wetland", "Mangroves", "Moss and lichen",
        "Pre-event Water", "Flood inundation", "River Ice"
    ]
    patchList = [mpatches.Patch(color=palette_MergeProductRGB[i], label=NameofLabel[i]) for i in range(len(NameofLabel))]
    return [patchList, cmap, norm]


def _wgs84_axis_ticks(transform, crs, nrows, ncols, n_ticks=7):
    """
    Map pixel column/row indices (imshow origin upper) to lon/lat for axis labels.
    Uses per-pixel projected coordinates + warp to EPSG:4326 — not linear steps in
    a hybrid affine (WGS84 corners + projected pixel size), which mislabels UTM rasters.
    """
    wgs84 = rasterio.crs.CRS.from_epsg(4326)
    if transform is None or crs is None or nrows < 1 or ncols < 1:
        return None
    n_ticks = max(2, int(n_ticks))
    xticks = np.linspace(0, max(ncols - 1, 0), num=n_ticks, dtype=np.float64)
    yticks = np.linspace(0, max(nrows - 1, 0), num=n_ticks, dtype=np.float64)
    row_bottom = np.full(xticks.shape, nrows - 1, dtype=np.float64)
    xs, ys = transform_xy(transform, row_bottom, xticks, offset="center")
    lons, _ = warp_transform(crs, wgs84, xs, ys)
    col_left = np.zeros(yticks.shape, dtype=np.float64)
    xs2, ys2 = transform_xy(transform, yticks, col_left, offset="center")
    _, lats = warp_transform(crs, wgs84, xs2, ys2)

    def _fmt(v):
        if not np.isfinite(v):
            return ""
        return ("%.4f" % v).rstrip("0").rstrip(".")

    xlabs = [_fmt(lon) for lon in np.asarray(lons).ravel()]
    ylabs = [_fmt(lat) for lat in np.asarray(lats).ravel()]
    return xticks, xlabs, yticks, ylabs


def SIENA_quicklookshow(savepath, arr_list, Mergeresult_PLTpara, titleName, transform=None, crs=None):
    """Save the colored overview as JPEG (and optionally PNG)."""
    tarr1 = arr_list
    shape_arr1 = tarr1.shape
    if shape_arr1[0] > shape_arr1[1]:
        fig = plt.figure(figsize=(7, 7 * shape_arr1[0] / shape_arr1[1]))
    else:
        fig = plt.figure(figsize=(7 * shape_arr1[1] / shape_arr1[0], 7))
    ax = fig.add_subplot(111)
    nrows, ncols = int(shape_arr1[0]), int(shape_arr1[1])
    tick_info = _wgs84_axis_ticks(transform, crs, nrows, ncols, n_ticks=7)
    if tick_info is not None:
        xticks, xlabs, yticks, ylabs = tick_info
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabs, fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabs, fontsize=8)
        ax.set_xlabel("Longitude (°)", fontsize=9)
        ax.set_ylabel("Latitude (°)", fontsize=9)
    else:
        ax.set_xlabel("Column (pixel)", fontsize=9)
        ax.set_ylabel("Row (pixel)", fontsize=9)
    ax.imshow(tarr1, interpolation="nearest", cmap=Mergeresult_PLTpara[1], norm=Mergeresult_PLTpara[2])
    fig.legend(handles=Mergeresult_PLTpara[0], loc="upper left", bbox_to_anchor=(0.9, 0.88), ncol=1)
    fig.suptitle(titleName, fontsize=16)
    #plt.savefig(savepath[0], dpi=150, format="png", bbox_inches="tight")
    plt.savefig(savepath[1], dpi=150, format="jpeg", bbox_inches="tight")
    plt.close()


def png_look(flood_arr, dry_arr, dire_arr, WOP_arr, LCC_arr, Dirout, fname, src, valid_mask, permanent_water_wo_threshold=75):
    """Compose land cover + permanent water + flood inundation and write the JPEG overview."""
    arr_quicklook = LCC_arr.copy()
    # Display: permanent/pre-event water = WO >= threshold or LCC == 80 (water); rest of water = flood inundation
    mask_pw = np.logical_or(WOP_arr >= permanent_water_wo_threshold, LCC_arr == 80)
    arr_quicklook[mask_pw] = 80
    arr_quicklook[np.logical_and(flood_arr == 1, arr_quicklook != 80)] = 120
    if dry_arr is not None:
        arr_quicklook[np.logical_and(dry_arr == 1, LCC_arr != 80)] = 110
        fname = "3classes_" + fname
    else:
        fname = "2classes_" + fname
    if dire_arr is not None:
        liquidmask = np.logical_and(dire_arr != 1, mask_pw)
        liquidmask[~valid_mask] = False
        arr_quicklook[liquidmask] = 130
        fname = "RiverIce_" + fname
    PLT_plate = quicklook_color()
    savepath = [os.path.join(Dirout, "SIENA_" + fname + ".png"), os.path.join(Dirout, "SIENA_" + fname + ".jpg")]
    SIENA_quicklookshow(savepath, arr_quicklook, PLT_plate, fname, transform=src.transform, crs=src.crs)


def flood_per_pre_save(savepath_RGBfile, mask_profile, floodarr, woparr, LCCarr, prearr, liquidarr, valid_mask, permanent_water_wo_threshold=75):
    """Write the paletted SIENA_2classes_RGB GeoTIFF (permanent water vs flood inundation)."""
    output_arr = floodarr.copy().astype(np.float32)
    # Display: permanent water = WO >= threshold or LCC == 80; rest of water pixels = flood inundation
    mask_pw = np.logical_or(woparr >= permanent_water_wo_threshold, LCCarr == 80)
    floodmask = np.logical_and(floodarr == 1, ~mask_pw)
    if prearr is not None:
        floodmask = np.logical_and(floodmask, prearr != 1)
    output_arr[floodmask] = 3
    if prearr is not None:
        output_arr[np.logical_and(prearr == 1, ~mask_pw)] = 2
    output_arr[mask_pw] = 1
    if liquidarr is not None:
        liquidmask = np.logical_and(liquidarr == 0, mask_pw)
        liquidmask[~valid_mask] = False
        output_arr[liquidmask] = 4
    mask_profile = dict(mask_profile)
    mask_profile.update(dtype=rasterio.uint8, nodata=255)
    output_arr[~valid_mask] = mask_profile["nodata"]
    output_arr = output_arr.astype(mask_profile["dtype"])
    output_arr[output_arr == 0] = 255
    with rasterio.open(savepath_RGBfile, "w", **mask_profile, compress="deflate") as dst:
        dst.write(output_arr, indexes=1)
    ds = gdal.Open(savepath_RGBfile, 1)
    band = ds.GetRasterBand(1)
    ct = gdal.ColorTable()
    for idx, i in enumerate([1, 2, 3, 4]):
        ct.SetColorEntry(i, tuple([[0, 92, 230], [0, 197, 255], [219, 0, 0], [255, 215, 0]][idx]))
    band.SetRasterColorTable(ct)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    band.SetNoDataValue(255)
    band.WriteArray(output_arr)
    del band, ds


def quick_look(output):
    """Write SIENA_2classes_RGB GeoTIFF, JPEG overview, and copy SIENA_raw if needed."""
    output = composition_sim(output)
    flood_arr, dry_arr, dire_arr, WOP_arr, LCC_arr, valid_mask = purge2array_sim(output)
    Dirout = output[0][0].dirOut
    fname = output[0][0].granules
    src = output[0][0].ref_src
    src_meta = output[0][0].refmeta
    wo_th = getattr(output[0][0], "permanent_water_wo_threshold", 75)
    two_classes_RGBfile = os.path.join(Dirout, "SIENA_2classes_RGB_" + fname + ".tif")
    flood_per_pre_save(two_classes_RGBfile, src_meta, flood_arr, WOP_arr, LCC_arr, None, None, valid_mask, wo_th)
    png_look(flood_arr, None, None, WOP_arr, LCC_arr, Dirout, fname, src, valid_mask, wo_th)
    print("RGB done,", two_classes_RGBfile)
    # RiverIce products (raster TIF + PNG/JPG) disabled; uncomment if needed
    # if dire_arr is not None:
    #     Two_classes_RiverIce_RGBfile = os.path.join(Dirout, "SIENA_2classes_RiverIce_RGB_" + fname + ".tif")
    #     flood_per_pre_save(Two_classes_RiverIce_RGBfile, src_meta, flood_arr, WOP_arr, LCC_arr, None, dire_arr, valid_mask, wo_th)
    #     png_look(flood_arr, None, dire_arr, WOP_arr, LCC_arr, Dirout, fname, src, valid_mask, wo_th)
    if dry_arr is not None:
        Three_classes_RGBfile = os.path.join(Dirout, "SIENA_3classes_RGB_" + fname + ".tif")
        flood_per_pre_save(Three_classes_RGBfile, src_meta, flood_arr, WOP_arr, LCC_arr, dry_arr, None, valid_mask, wo_th)
        png_look(flood_arr, dry_arr, None, WOP_arr, LCC_arr, Dirout, fname, src, valid_mask, wo_th)
    src_comp = getattr(output[0][0], "fileOutCompensate", "")
    dest_raw = os.path.join(Dirout, "SIENA_raw_" + fname + ".tif")
    if src_comp and os.path.abspath(src_comp) != os.path.abspath(dest_raw):
        shutil.copy(src_comp, dest_raw)
    return two_classes_RGBfile
