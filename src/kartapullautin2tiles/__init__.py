from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import imagesize
import mercantile
import numpy as np
import pyproj
import rasterio.merge
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

__version__ = version("kartapullautin2tiles")


def _load_img(pgw_file: Path):
    """Load pgw file to geopandas row"""
    # Parse parameters from the .pgw file
    # The file format is:
    # line 1: pixel X size
    # line 2: rotation term for Y (usually 0)
    # line 3: rotation term for X (usually 0)
    # line 4: pixel Y size (usually negative)
    # line 5: X coordinate of top-left corner (as per user interpretation)
    # line 6: Y coordinate of top-left corner (as per user interpretation)
    ps_x, _, _, ps_y, x, y = pgw_file.read_text().strip().split("\n")
    ps_x, ps_y, x, y = float(ps_x), float(ps_y), float(x), float(y)

    img_file = pgw_file.with_suffix(".png")
    img_width, img_height = imagesize.get(img_file)

    tile_total_width_map_units = img_width * ps_x
    tile_total_height_map_units = img_height * ps_y

    tile_polygon = Polygon(
        [
            (x, y),  # Top-left
            (x + tile_total_width_map_units, y),  # Top-right
            (x + tile_total_width_map_units, y + tile_total_height_map_units),  # Bottom-right
            (x, y + tile_total_height_map_units),  # Bottom-left
        ]
    )

    return {"id": pgw_file.stem, "pgw_file": pgw_file, "img_file": img_file, "geometry": tile_polygon}


def load_kartapullautin_dir(dir: Path, *, proj="EPSG:25832", pattern="*depr*.pgw"):
    """
    Load coordinates from pgw files into a GeoPandas data frame

    Parameters
    ----------
    dir
        input directory (kartapullautin output dir)
    proj
        EPSG string of the projection used
    pattern
        search pattern for the output folder
    """
    return gpd.GeoDataFrame((_load_img(f) for f in dir.glob(pattern)), crs=proj)


def _get_tile_bb(tile: mercantile.Tile, crs: str):
    """Get rows from the dataframe that overlap with tile"""
    tile_wgs84_bounds = mercantile.bounds(tile)  # Get the WGS84 bounds of the selected web mercator tile

    # EPSG:4326 = WGS84
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    minx, miny = transformer.transform(*tile_wgs84_bounds[:2])
    maxx, maxy = transformer.transform(*tile_wgs84_bounds[2:])
    return (minx, miny, maxx, maxy)


def stitch_tile(gpdf: gpd.GeoDataFrame, tile: mercantile.Tile, *, tile_size: int = 256) -> Image.Image:
    """
    Create a single Mercator tile from one or multiple kartapullautin tiles

    (there's no guarantee that a tile is always fully contained in one kartapullautin tile, especially
    on lower zoom levels.)
    """
    query_polygon = _get_tile_bb(tile, str(gpdf.crs))
    tmp_df = gpdf.loc[gpdf.intersects(Polygon.from_bounds(*query_polygon))].reset_index()
    if tmp_df.empty:
        print("No intersecting tiles found. Displaying a blank image.")
        return Image.new("RGB", (tile_size, tile_size), color="white")
    else:
        # query_bounds = tuple(tmp_df.total_bounds)
        mosaic_array, _ = rasterio.merge.merge(
            tmp_df["img_file"],
            bounds=query_polygon,
            nodata=255,  # Use 0 as the nodata value for areas not covered
            dtype=np.uint8,  # Assuming 8-bit PNGs; adjust if necessary
        )

        # Convert the NumPy array from rasterio (bands, height, width) to a PIL Image
        num_bands = mosaic_array.shape[0]

        if num_bands == 1:
            # Grayscale image: mosaic_array is (1, H, W)
            image_data_for_pil = mosaic_array[0]  # Get (H, W)
            mode = "L"
        elif num_bands == 3:
            # RGB image: transpose from (3, H, W) to (H, W, 3)
            image_data_for_pil = np.transpose(mosaic_array, (1, 2, 0))
            mode = "RGB"
        elif num_bands == 4:
            # RGBA image: transpose from (4, H, W) to (H, W, 4)
            image_data_for_pil = np.transpose(mosaic_array, (1, 2, 0))
            mode = "RGBA"
        else:
            raise ValueError(f"Unsupported number of bands in mosaic: {num_bands}. Expected 1, 3, or 4.")

        pil_image = Image.fromarray(image_data_for_pil, mode=mode)
        return pil_image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)


def make_tiles(gpdf: gpd.GeoDataFrame, *, out_dir: Path, min_zoom: int = 10, max_zoom: int = 18):
    """
    Create a tile directory

    Parameters
    ----------
    gpdf
        GeoPandas DataFrame
    out_dir
        output directory for tiles (z/x/y folder structure)
    """
    transformer_to_wgs84 = pyproj.Transformer.from_crs(gpdf.crs, "EPSG:4326", always_xy=True)

    # overall bounding box in EPSG:4326
    west_lon, south_lat = transformer_to_wgs84.transform(*gpdf.total_bounds[:2])
    east_lon, north_lat = transformer_to_wgs84.transform(*gpdf.total_bounds[2:])

    for tile in tqdm(mercantile.tiles(west_lon, south_lat, east_lon, north_lat, zooms=range(min_zoom, max_zoom + 1))):
        img = stitch_tile(gpdf, tile)
        tile_path = out_dir / str(tile.z) / str(tile.x)
        tile_path.mkdir(parents=True, exist_ok=True)
        img.save(tile_path / f"{tile.y}.png")
