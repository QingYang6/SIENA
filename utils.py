"""Raster helpers: read/write, reproject onto a reference grid, mosaic, timers."""
import os
import time
import psutil
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box
from pyproj import Transformer

class Timer(object):
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.tstart = time.time()
    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def tim(func):
    """Print wall-clock time for a function call."""
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}, before:{:,}; After:{:,};".format(
            func.__name__, mem_after - mem_before,mem_before,mem_after))
        return result
    return wrapper

def reproject_clip_readsrc(in_rasfile,ref):
    with rasterio.open(in_rasfile) as src:
        in_ras = src.read()
        src_meta = src.meta.copy()
        out_arr = reproject_clip_direct(in_ras,src_meta,ref)
    return out_arr

def reproject_clip_direct(in_ras,src,ref):
    """Warp an array from src metadata onto a reference grid (nearest neighbor)."""
    dest_shape = (src['count'], ref['height'], ref['width'])
    out_arr = np.zeros(dest_shape, dtype = src['dtype'])
    rasterio.warp.reproject(
        source=in_ras,
        destination=out_arr,
        src_transform=src['transform'],
        src_crs=src['crs'],
        dst_transform=ref['transform'],
        dst_crs=ref['crs'],
        resampling=rasterio.warp.Resampling.nearest,
        src_nodata=src.get('nodata'),
        dst_nodata=src.get('nodata'),
        )
    return out_arr


# --- merged from utils_merge ---
def write_color_raster(raster_d_m, output_file, colormap):
    with rasterio.open(output_file, 'w', **raster_d_m['meta']) as dst:
        dst.write(raster_d_m['arr'], 1)
        dst.write_colormap(1, colormap)

def getfile_list(folder_path, extension='.tif'):
    tif_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                tif_files.append(os.path.join(root, file))
    return tif_files

def intersecting_files(file_list, bounds):
    intersecting_files_list = []
    input_polygon = box(*bounds)
    for file_path in file_list:
        with rasterio.open(file_path) as src:
            raster_bounds = src.bounds
            if src.crs != rasterio.crs.CRS.from_epsg(4326):
                raster_bounds = transform_bounds(src.crs, "EPSG:4326", *raster_bounds)
            raster_polygon = box(*raster_bounds)
            if input_polygon.intersects(raster_polygon):
                intersecting_files_list.append(file_path)
    return intersecting_files_list

def getinterfolder_files(folder_path, bounds, meta, extension='.tif'):
    if not meta['crs'].to_epsg() == 4326:
        bounds_wgs84 = transform_bounds(meta['crs'], "EPSG:4326", *bounds)
    else:
        bounds_wgs84 = bounds
    tif_files_list = getfile_list(folder_path, extension)
    intersect_list = intersecting_files(tif_files_list, bounds_wgs84)
    return intersect_list

def getcropfolder_files(folder_path, bounds, meta, crop_index = True ,extension='.tif'):
    if not meta['crs'].to_epsg() == 4326:
        bounds_wgs84 = transform_bounds(meta['crs'], "EPSG:4326", *bounds)
    else:
        bounds_wgs84 = bounds
    tif_files_list = getfile_list(folder_path, extension)
    intersect_list = intersecting_files(tif_files_list, bounds_wgs84)
    intersect_merged = mergelist(intersect_list)
    if crop_index:
        intersect_merged_crop = cropfile(intersect_merged, bounds_wgs84)
        return read_raster(intersect_merged_crop)
    else:
        return read_raster(intersect_merged)

def setnan_writeraster(flood_depth_result,depth_meta,out_bounds,depth_output):
    depth_meta.update({'dtype': 'float32','nodata': 0})
    depth_d_m = create_raster_data_dict(flood_depth_result, depth_meta, out_bounds)
    depth_d_m = set_nan(depth_d_m)
    write_raster(depth_d_m, depth_output)

def set_nan(input_data_and_metadata):
    input_data_and_metadata['arr'][np.isnan(input_data_and_metadata['arr'])] = None
    input_data_and_metadata['arr'][input_data_and_metadata['arr']==0] = None
    input_data_and_metadata['meta'].update({'dtype': 'float32','nodata': None})
    return input_data_and_metadata

def create_raster_data_dict(arr, metadata, bounds):
    """Pack array, rasterio metadata, and bounds into the dict used across SIENA."""
    return {
        'arr': arr,
        'meta': metadata,
        'bounds': bounds
    }

def sample_raster_at_latlon(raster_data_dict, lat_2d, lon_2d, nodata_out=0.0):
    """Sample a raster at 2D lat/lon points (for optional NetCDF native-grid ancillary)."""
    from rasterio.transform import rowcol
    arr = raster_data_dict["arr"]
    meta = raster_data_dict["meta"]
    h, w = arr.shape
    transform = meta["transform"]
    crs = meta.get("crs")
    if crs and crs != rasterio.crs.CRS.from_epsg(4326):
        from rasterio.warp import transform as transform_coords
        xs, ys = transform_coords("EPSG:4326", crs, lon_2d.ravel(), lat_2d.ravel())
        x_pts = np.asarray(xs).reshape(lat_2d.shape)
        y_pts = np.asarray(ys).reshape(lat_2d.shape)
    else:
        x_pts, y_pts = lon_2d, lat_2d
    rows, cols = rowcol(transform, x_pts.ravel(), y_pts.ravel())
    rows = np.rint(rows).astype(np.int32).reshape(lat_2d.shape)
    cols = np.rint(cols).astype(np.int32).reshape(lat_2d.shape)
    out = np.full(lat_2d.shape, nodata_out, dtype=np.float32)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out[valid] = arr[rows[valid], cols[valid]]
    return out

def reproject_clip(input_binary, input_reference, resampling_method=Resampling.nearest):
    """Reproject one raster-data dict onto another's grid."""
    src_meta = input_binary['meta']
    ref_meta = input_reference['meta']
    ref_height = ref_meta['height']
    ref_width = ref_meta['width']
    kwargs = src_meta.copy()
    kwargs.update({
        'crs': ref_meta['crs'],
        'transform': ref_meta['transform'],
        'width': ref_width,
        'height': ref_height,
        'compress': 'DEFLATE'})
    src_data = input_binary['arr']
    ref_data = input_reference['arr']
    output_arr = np.zeros_like(ref_data)
    reproject(
        source=src_data,
        destination=output_arr,
        src_transform=src_meta['transform'],
        src_crs=src_meta['crs'],
        dst_transform=ref_meta['transform'],
        dst_crs=ref_meta['crs'],
        resampling=resampling_method
    )
    return create_raster_data_dict(output_arr, kwargs, input_reference['bounds'])

def read_raster(file_path):
    """Read a GeoTIFF into a raster-data dict (array, meta, bounds)."""
    with rasterio.open(file_path) as src:
        src_data = src.read(1).astype(np.float32)
        metadata = src.meta.copy()
        bounds = src.bounds
    return create_raster_data_dict(src_data, metadata, bounds)

def write_raster(raster_data, output_file_path,write_band=1):
    """Write a raster-data dict to a GeoTIFF."""
    arr = raster_data['arr']
    metadata = raster_data['meta']
    metadata.update({'compress': 'DEFLATE'})
    with rasterio.open(output_file_path, "w", **metadata) as dest:
        dest.write(arr,write_band)

def calculate_pixel_size(raster_data_dict):
    metadata = raster_data_dict['meta']
    transform = metadata['transform']
    pixel_width = transform[0]
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

def mergelist(rasters_to_merge):
    if len(rasters_to_merge) == 1:
        return rasters_to_merge[0]
    src_files_to_mosaic = [rasterio.open(raster) for raster in rasters_to_merge]
    try:
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
    finally:
        for src in src_files_to_mosaic:
            try: src.close()
            except: pass
    out_mem = MemoryFile()
    with rasterio.open(out_mem, 'w', **out_meta) as dest:
        dest.write(mosaic)
    out_mem.seek(0)
    return out_mem

def reproject_withsrc(in_ras, src, ref):
    out_mem = MemoryFile()
    with out_mem.open(**ref) as dst:
        for i in range(1, src.count + 1):
            rasterio.warp.reproject(
                source=in_ras,
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref['transform'],
                dst_crs=ref['crs'],
                resampling=rasterio.warp.Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                copy_src_overviews=True)
    out_mem.seek(0)
    return out_mem

def cropfile(input_file,bounds):
    new_bounds = box(*bounds)
    with rasterio.open(input_file) as src:
        out_image, out_transform = mask(src, [new_bounds], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
    out_mem = MemoryFile()
    with rasterio.open(out_mem, "w", **out_meta) as dest:
        dest.write(out_image)
    out_mem.seek(0)
    return out_mem

def vectorized_f1_scores_matrix(candi_FIM_3D, full_FIM, lcc_arr, labels, weights):
    weight_matrix = np.ones_like(lcc_arr)
    for label, weight in zip(labels, weights):
        weight_matrix[lcc_arr == label] = weight
    TP = np.sum(weight_matrix[:, :, np.newaxis] * (candi_FIM_3D == 1) * (full_FIM[:, :, np.newaxis] == 1), axis=(0, 1))
    FP = np.sum(weight_matrix[:, :, np.newaxis] * (candi_FIM_3D == 1) * (full_FIM[:, :, np.newaxis] == 0), axis=(0, 1))
    FN = np.sum(weight_matrix[:, :, np.newaxis] * (candi_FIM_3D == 0) * (full_FIM[:, :, np.newaxis] == 1), axis=(0, 1))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0
    return f1

def vectorized_f1_scores_vector(candi_FIM, full_FIM, lcc_arr, labels, weights):
    TP = 0
    FP = 0
    FN = 0
    for label in labels:
        label_indices = np.where(lcc_arr == label)[0]
        if len(label_indices) > 0:
            label_weight = weights[labels.index(label)]
            TP += np.sum(label_weight * (candi_FIM[label_indices] == 1) * (full_FIM[label_indices] == 1))
            FP += np.sum(label_weight * (candi_FIM[label_indices] == 1) * (full_FIM[label_indices] == 0))
            FN += np.sum(label_weight * (candi_FIM[label_indices] == 0) * (full_FIM[label_indices] == 1))
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall    = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return f1

def convert_bounds(src_crs, src_bounds):
    """Convert a bounding box to WGS84 lon/lat."""
    return transform_bounds(src_crs, "EPSG:4326", *src_bounds)
