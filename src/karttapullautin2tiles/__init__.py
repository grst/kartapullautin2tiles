import importlib.resources
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import imagesize
import mercantile
import numpy as np
import pyproj
import rasterio.merge
import rasterio.transform
import rasterio.warp
from PIL import Image
from pyproj.crs.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

__version__ = version("karttapullautin2tiles")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

WGS_84_CRS = "EPSG:4326"


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


def _get_tiles_center(tiles: Sequence[mercantile.Tile]) -> tuple[float, float] | None:
    """Get the center of a set of tiles in Lng/Lat coordinates (web mercator projection)"""
    bboxes = [mercantile.bounds(t) for t in tiles]

    if not bboxes:
        return None

    min_x = min(bbox.west for bbox in bboxes)
    max_x = max(bbox.east for bbox in bboxes)
    min_y = min(bbox.south for bbox in bboxes)
    max_y = max(bbox.north for bbox in bboxes)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    return center_x, center_y


def load_karttapullautin_dir(dir: Path, *, proj: str | CRS = "EPSG:25832", pattern="*depr*.pgw") -> gpd.GeoDataFrame:
    """
    Load coordinates from pgw files into a GeoPandas data frame

    Parameters
    ----------
    dir
        input directory (karttapullautin output dir)
    proj
        EPSG string of the projection used
    pattern
        search pattern for the output folder
    """
    files = list(dir.glob(pattern))
    if not files:
        # Return empty GeoDataFrame with correct columns and CRS
        return gpd.GeoDataFrame(columns=["id", "pgw_file", "img_file", "geometry"], crs=proj)
    return gpd.GeoDataFrame((_load_img(f) for f in files), crs=proj)


def _get_tile_bb_polygon(tile: mercantile.Tile, crs: str | CRS):
    """Get the polygon that refers to the boundaries of `tile`, transformed to `crs`"""
    tile_wgs84_bounds = mercantile.bounds(tile)
    tile_wgs84_polygon = Polygon.from_bounds(*tile_wgs84_bounds)

    # Transform the polygon to the gpdf CRS
    transformer = pyproj.Transformer.from_crs(WGS_84_CRS, crs, always_xy=True)
    transformed_coords = []
    for x, y in tile_wgs84_polygon.exterior.coords:
        transformed_x, transformed_y = transformer.transform(x, y)
        transformed_coords.append((transformed_x, transformed_y))

    return Polygon(transformed_coords)


def _subset_array(
    array: np.ndarray, transform: rasterio.Affine, *, array_crs: str | CRS, tile: mercantile.Tile
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Subset a merged image array to (approximately) the area of the tile of interest.

    This is done for increased performance of rasterio.warp.reproject.

    Parameters
    ----------
    array
        merge image; output of rasterio.merge.merge
    array_crs
        The CRS used by the image in `array`
    tile
        The tile of interest
    """
    tile_wgs84_bounds = mercantile.bounds(tile)
    # Transform tile bounds to source CRS to determine subset bounds
    transformer = pyproj.Transformer.from_crs(WGS_84_CRS, array_crs, always_xy=True)
    src_minx, src_miny = transformer.transform(*tile_wgs84_bounds[:2])
    src_maxx, src_maxy = transformer.transform(*tile_wgs84_bounds[2:])

    # Add some padding to ensure we capture the full tile area after reprojection
    padding_factor = 0.1
    width_padding = (src_maxx - src_minx) * padding_factor
    height_padding = (src_maxy - src_miny) * padding_factor

    src_minx -= width_padding
    src_miny -= height_padding
    src_maxx += width_padding
    src_maxy += height_padding

    # Convert source bounds to pixel coordinates
    inv_transform = ~transform
    left_col, top_row = inv_transform * (src_minx, src_maxy)  # type: ignore
    right_col, bottom_row = inv_transform * (src_maxx, src_miny)  # type: ignore

    # Ensure we stay within array bounds
    left_col = max(0, int(np.floor(left_col)))
    top_row = max(0, int(np.floor(top_row)))
    right_col = min(array.shape[-1], int(np.ceil(right_col)))
    bottom_row = min(array.shape[-2], int(np.ceil(bottom_row)))

    # Check if we have valid dimensions
    if left_col >= right_col or top_row >= bottom_row:
        raise ValueError("Tile does not overlap with input array")

    subset_array = array[:, top_row:bottom_row, left_col:right_col]

    # Calculate the actual bounds that correspond to the clipped pixel coordinates
    # This ensures the transform matches the actual subset array dimensions
    actual_minx, actual_maxy = transform * (left_col, top_row)  # type: ignore
    actual_maxx, actual_miny = transform * (right_col, bottom_row)  # type: ignore

    # Create transform for the subset using actual bounds
    subset_transform = rasterio.transform.from_bounds(
        actual_minx, actual_miny, actual_maxx, actual_maxy, right_col - left_col, bottom_row - top_row
    )

    return subset_array, subset_transform


def extract_and_transform_tile(
    array: np.ndarray, transform: rasterio.Affine, tile: mercantile.Tile, *, src_crs: str | CRS, tile_size=256
):
    """
    Extract the image that corresponds to a specific tile from an array and display it.

    Properly reprojects the image data from the source CRS to WGS84 (Web Mercator tile CRS).

    Parameters
    ----------
    array : numpy.ndarray
        The merged raster array from rasterio.merge.merge()
    transform : affine.Affine
        The affine transform from rasterio.merge.merge()
    tile : mercantile.Tile
        The tile to extract
    source_crs : str
        The coordinate reference system of the source array (default: "EPSG:25832")
    tile_size : int
        Target size for the output tile (default: 256)

    Returns
    -------
    PIL.Image.Image
        The extracted, reprojected, and resized tile image
    """
    # Get tile bounds in WGS84
    tile_wgs84_bounds = mercantile.bounds(tile)

    # Create empty tile
    dst_array = np.full((3, tile_size, tile_size), dtype=array.dtype, fill_value=255)

    try:
        subset_array, subset_transform = _subset_array(array, transform, array_crs=src_crs, tile=tile)

        # Create a transform for the destination tile
        dst_transform = from_bounds(*tile_wgs84_bounds, tile_size, tile_size)

        # Reproject the subset data
        rasterio.warp.reproject(
            source=subset_array,
            destination=dst_array,
            src_transform=subset_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=WGS_84_CRS,
            dst_nodata=255,
            resampling=rasterio.warp.Resampling.lanczos,
        )
    except ValueError:
        pass

    image_data = np.transpose(dst_array, (1, 2, 0))
    # Create PIL image
    pil_image = Image.fromarray(image_data.astype(np.uint8), mode="RGB")

    return pil_image


def list_tiles(dir: Path, *, proj: str = "EPSG:25832", pattern="*depr*.pgw", min_zoom: int = 12):
    """
    List the tiles that are covered by the karttapullautin directory at the given zoom level.

    Use this as a list of tiles that can be passed to make_tiles.

    Parameters
    ----------
    dir
        Input directory (karttapullautin output dir)
    proj
        EPSG string of the projection used
    pattern
        Search pattern for the pgw files
    min_zoom
        Zoom level to generate tiles for

    Returns
    -------
    Generator of tiles that cover the bounding box of all images in the directory
    """
    gpdf = load_karttapullautin_dir(dir, proj=proj, pattern=pattern)

    transformer_to_wgs84 = pyproj.Transformer.from_crs(gpdf.crs, WGS_84_CRS, always_xy=True)

    if not gpdf.shape[0]:
        logging.info("No tiles found.")
        return []

    # overall bounding box in EPSG:4326
    west_lon, south_lat = transformer_to_wgs84.transform(*gpdf.total_bounds[:2])
    east_lon, north_lat = transformer_to_wgs84.transform(*gpdf.total_bounds[2:])

    return mercantile.tiles(west_lon, south_lat, east_lon, north_lat, zooms=[min_zoom])


def make_tiles(
    in_dir: Path,
    out_dir: Path,
    tiles: Sequence[mercantile.Tile],
    *,
    proj: str = "EPSG:25832",
    pattern="*depr*.pgw",
    max_zoom: int = 17,
):
    """
    Create a tile directory from karttapullautin output.

    Note that all images required for a tile at min_zoom need to fit in memory. If you have
    memory issues, consider setting a higher zoom level.

    Parameters
    ----------
    in_dir
        Input directory containing karttapullautin output files
    out_dir
        Output directory for tiles (z/x/y folder structure)
    tiles
        Sequence of tiles to process at the minimum zoom level
    proj
        EPSG string of the projection used by input images
    pattern
        Search pattern for pgw files in the input directory
    max_zoom
        Maximum zoom level to generate tiles for
    """
    gpdf = load_karttapullautin_dir(in_dir, proj=proj, pattern=pattern)
    assert gpdf.crs is not None

    # iterate over min zoom tiles
    for parent_tile in tiles:
        logger.info(f"Working on {parent_tile}")

        query_polygon = _get_tile_bb_polygon(parent_tile, gpdf.crs)
        tmp_df = gpdf.loc[gpdf.intersects(query_polygon)].reset_index()
        if not tmp_df.shape[0]:
            logger.info("Empty tile, skipping")
            continue

        # stitch tiles
        img_array, img_transform = rasterio.merge.merge(
            tmp_df["img_file"], bounds=tuple(tmp_df.total_bounds), nodata=255, dtype=np.uint8
        )
        for zoom in range(parent_tile.z, max_zoom + 1):
            for tile in mercantile.children(parent_tile, zoom=zoom):
                img = extract_and_transform_tile(img_array, img_transform, tile, src_crs=gpdf.crs)

                tile_path = out_dir / str(tile.z) / str(tile.x)
                tile_path.mkdir(parents=True, exist_ok=True)
                img.save(tile_path / f"{tile.y}.png")


def get_html_viewer(lon_center: float, lat_center: float, *, default_zoom: int, min_zoom: int, max_zoom: int):
    """
    Output the HTML for a viewer application that can preview tiles in the same folder.

    Parameters
    ----------
    lon_center
        longitude coordinate of the default center of the map
    lat_center
        latitude coordinate of the default center of the map
    default_zoom
        default zoom
    """
    html_template = (importlib.resources.files("karttapullautin2tiles.assets") / "local_tiles_viewer.html").read_text(
        "utf-8"
    )

    html = html_template.replace("{{lon_center}}", str(lon_center))
    html = html.replace("{{lat_center}}", str(lat_center))
    html = html.replace("{{default_zoom}}", str(default_zoom))
    html = html.replace("{{min_zoom}}", str(min_zoom))
    html = html.replace("{{max_zoom}}", str(max_zoom))

    return html
