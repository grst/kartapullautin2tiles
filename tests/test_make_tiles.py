import tempfile
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
import pytest
import rasterio
import rasterio.transform
from PIL import Image
from pyproj.crs.crs import CRS
from shapely.geometry import Polygon

import karttapullautin2tiles


def test_package_has_version():
    assert karttapullautin2tiles.__version__ is not None


# Tests for _load_img function
def test_load_img_with_valid_pgw(sample_pgw_file):
    """Test loading a valid PGW file"""
    result = karttapullautin2tiles._load_img(sample_pgw_file)

    assert result["id"] == "576_5265.laz_depr"
    assert result["pgw_file"] == sample_pgw_file
    assert result["img_file"] == sample_pgw_file.with_suffix(".png")
    assert isinstance(result["geometry"], Polygon)

    # Check that the polygon has 5 coordinates (closed polygon)
    coords = list(result["geometry"].exterior.coords)
    assert len(coords) == 5
    assert coords[0] == coords[-1]  # First and last should be the same (closed)


def test_load_img_pgw_parsing(temp_image_files):
    """Test that PGW file parameters are parsed correctly"""
    result = karttapullautin2tiles._load_img(temp_image_files["pgw_file"])

    # Check the geometry bounds match expected values
    # With pixel size 1.0, -1.0 and 10x10 image starting at (100, 200)
    bounds = result["geometry"].bounds
    assert bounds[0] == 100.0  # minx
    assert bounds[1] == 190.0  # miny (200 + 10 * -1.0)
    assert bounds[2] == 110.0  # maxx (100 + 10 * 1.0)
    assert bounds[3] == 200.0  # maxy


# Tests for _get_tiles_center function
def test_get_tiles_center_basic():
    """Test getting center of a set of tiles"""
    tiles = [
        mercantile.Tile(0, 0, 1),
        mercantile.Tile(1, 0, 1),
        mercantile.Tile(0, 1, 1),
        mercantile.Tile(1, 1, 1),
    ]

    result = karttapullautin2tiles._get_tiles_center(tiles)

    assert result is not None
    lon, lat = result
    assert isinstance(lon, float)
    assert isinstance(lat, float)
    # Center should be around 0,0 for these tiles
    assert abs(lon) < 1
    assert abs(lat) < 1


def test_get_tiles_center_empty():
    """Test getting center of empty tile list"""
    result = karttapullautin2tiles._get_tiles_center([])
    assert result is None


def test_get_tiles_center_single_tile():
    """Test getting center of a single tile"""
    tile = mercantile.Tile(100, 50, 10)
    result = karttapullautin2tiles._get_tiles_center([tile])

    assert result is not None
    lon, lat = result

    # For a single tile, center should be the tile's center
    bounds = mercantile.bounds(tile)
    expected_lon = (bounds.west + bounds.east) / 2
    expected_lat = (bounds.south + bounds.north) / 2

    assert lon == pytest.approx(expected_lon)
    assert lat == pytest.approx(expected_lat)


# Tests for load_karttapullautin_dir function
def test_load_karttapullautin_dir_default_params(test_data_dir):
    """Test loading directory with default parameters"""
    result = karttapullautin2tiles.load_karttapullautin_dir(test_data_dir)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:25832"
    assert len(result) == 6  # Should find 6 PGW files
    assert all(col in result.columns for col in ["id", "pgw_file", "img_file", "geometry"])


@pytest.mark.parametrize(
    "proj,pattern,expected_count",
    [
        ("EPSG:25832", "*depr*.pgw", 6),
        ("EPSG:4326", "*.pgw", 6),
        (CRS.from_epsg(25832), "*depr*.pgw", 6),
    ],
)
def test_load_karttapullautin_dir_params(test_data_dir, proj, pattern, expected_count):
    """Test loading directory with different parameters"""
    result = karttapullautin2tiles.load_karttapullautin_dir(test_data_dir, proj=proj, pattern=pattern)

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == expected_count

    if isinstance(proj, str):
        assert result.crs == proj
    else:
        assert result.crs is not None
        assert result.crs.to_epsg() == proj.to_epsg()


def test_load_karttapullautin_dir_empty_pattern(test_data_dir):
    """Test loading directory with pattern that matches no files"""
    result = karttapullautin2tiles.load_karttapullautin_dir(test_data_dir, pattern="nonexistent*.pgw")

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


# Tests for list_tiles function
def test_list_tiles_basic(test_data_dir):
    """Test basic tile listing functionality"""
    tiles = list(karttapullautin2tiles.list_tiles(test_data_dir, min_zoom=12))

    assert len(tiles) > 0
    assert all(isinstance(tile, mercantile.Tile) for tile in tiles)
    assert all(tile.z == 12 for tile in tiles)


def test_list_tiles_empty_directory():
    """Test list_tiles with empty directory"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiles = list(karttapullautin2tiles.list_tiles(Path(tmp_dir), min_zoom=12))
        assert len(tiles) == 0


@pytest.mark.parametrize("min_zoom", [8, 10, 12, 15])
def test_list_tiles_different_zoom_levels(test_data_dir, min_zoom):
    """Test list_tiles with different zoom levels"""
    tiles = list(karttapullautin2tiles.list_tiles(test_data_dir, min_zoom=min_zoom))

    if len(tiles) > 0:  # Only check if tiles were found
        assert all(tile.z == min_zoom for tile in tiles)


def test_list_tiles_custom_params(test_data_dir):
    """Test list_tiles with custom projection and pattern"""
    tiles = list(karttapullautin2tiles.list_tiles(test_data_dir, proj="EPSG:25832", pattern="*depr*.pgw", min_zoom=10))

    assert len(tiles) > 0
    assert all(tile.z == 10 for tile in tiles)


# Tests for _get_tile_bb_polygon function
@pytest.mark.parametrize(
    "tile_z,tile_x,tile_y,crs",
    [
        (10, 500, 300, "EPSG:25832"),
        (15, 16000, 10000, "EPSG:4326"),
        (8, 100, 80, CRS.from_epsg(3857)),
    ],
)
def test_get_tile_bb_various_tiles(tile_z, tile_x, tile_y, crs):
    """Test getting bounding box polygon for various tiles and CRS"""
    tile = mercantile.Tile(tile_x, tile_y, tile_z)

    result = karttapullautin2tiles._get_tile_bb_polygon(tile, crs)

    assert isinstance(result, Polygon)
    bounds = result.bounds
    minx, miny, maxx, maxy = bounds
    assert minx < maxx
    assert miny < maxy
    assert all(isinstance(x, float) for x in bounds)


def test_get_tile_bb_consistency():
    """Test that tile bounding box polygon is consistent with mercantile bounds"""
    tile = mercantile.Tile(1000, 600, 12)
    crs = "EPSG:4326"

    result = karttapullautin2tiles._get_tile_bb_polygon(tile, crs)
    original_bounds = mercantile.bounds(tile)

    # For EPSG:4326, the bounds should be the same as mercantile.bounds
    bounds = result.bounds
    assert bounds[0] == pytest.approx(original_bounds.west)
    assert bounds[1] == pytest.approx(original_bounds.south)
    assert bounds[2] == pytest.approx(original_bounds.east)
    assert bounds[3] == pytest.approx(original_bounds.north)


# Tests for _subset_array function
def test_subset_array_basic(test_array):
    """Test basic array subsetting functionality"""
    transform = rasterio.transform.from_bounds(0, 0, 100, 100, 100, 100)
    tile = mercantile.Tile(0, 0, 1)  # Simple tile

    subset_array, subset_transform = karttapullautin2tiles._subset_array(
        test_array, transform, array_crs="EPSG:4326", tile=tile
    )

    assert subset_array.shape[0] == 3  # Same number of channels
    assert subset_array.shape[1] <= test_array.shape[1]  # Height should be <= original
    assert subset_array.shape[2] <= test_array.shape[2]  # Width should be <= original
    assert isinstance(subset_transform, rasterio.Affine)


def test_subset_array_no_overlap_raises_error():
    """Test that subsetting with no overlap raises ValueError"""
    # Create array covering coordinates 1000-2000, 1000-2000
    array = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)
    transform = rasterio.transform.from_bounds(1000, 1000, 2000, 2000, 100, 100)
    # Tile covering coordinates around 0,0 (no overlap)
    tile = mercantile.Tile(0, 0, 1)

    with pytest.raises(ValueError, match="Tile does not overlap with input array"):
        karttapullautin2tiles._subset_array(array, transform, array_crs="EPSG:4326", tile=tile)


def test_subset_array_edge_cases():
    """Test edge cases for array subsetting"""
    # Very small array
    array = np.ones((3, 5, 5), dtype=np.uint8)
    transform = rasterio.transform.from_bounds(-1, -1, 1, 1, 5, 5)
    tile = mercantile.Tile(0, 0, 1)

    subset_array, subset_transform = karttapullautin2tiles._subset_array(
        array, transform, array_crs="EPSG:4326", tile=tile
    )

    assert subset_array.shape[0] == 3
    assert subset_array.size > 0  # Should have some data


# Tests for extract_and_transform_tile function
def test_extract_and_transform_tile_basic(test_array):
    """Test basic tile extraction and transformation"""
    transform = rasterio.transform.from_bounds(-180, -85, 180, 85, 100, 100)
    tile = mercantile.Tile(0, 0, 1)

    result = karttapullautin2tiles.extract_and_transform_tile(test_array, transform, tile, src_crs="EPSG:4326")

    assert isinstance(result, Image.Image)
    assert result.size == (256, 256)  # Default tile size
    assert result.mode == "RGB"


@pytest.mark.parametrize("tile_size", [128, 256, 512])
def test_extract_and_transform_tile_sizes(tile_size):
    """Test tile extraction with different tile sizes"""
    array = np.random.randint(0, 255, (3, 50, 50), dtype=np.uint8)
    transform = rasterio.transform.from_bounds(-1, -1, 1, 1, 50, 50)
    tile = mercantile.Tile(0, 0, 1)

    result = karttapullautin2tiles.extract_and_transform_tile(
        array, transform, tile, src_crs="EPSG:4326", tile_size=tile_size
    )

    assert result.size == (tile_size, tile_size)


def test_extract_and_transform_tile_no_overlap():
    """Test tile extraction when there's no overlap (should return white tile)"""
    # Array covering a different area than the tile
    array = np.random.randint(0, 255, (3, 50, 50), dtype=np.uint8)
    transform = rasterio.transform.from_bounds(1000, 1000, 2000, 2000, 50, 50)
    tile = mercantile.Tile(0, 0, 1)  # Tile around 0,0

    result = karttapullautin2tiles.extract_and_transform_tile(array, transform, tile, src_crs="EPSG:4326")

    # Should return a white (255, 255, 255) tile when no data overlaps
    assert isinstance(result, Image.Image)
    assert result.size == (256, 256)
    # Check if most pixels are white (255) - allowing for some edge effects
    pixel_array = np.array(result)
    white_pixels = np.sum(pixel_array == 255)
    total_pixels = pixel_array.size
    assert white_pixels / total_pixels > 0.9  # Most pixels should be white


# Tests for make_tiles function (updated signature)
def test_make_tiles_basic(test_data_dir):
    """Test basic tile generation with new signature"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        # Get tiles using list_tiles function
        tiles = list(karttapullautin2tiles.list_tiles(test_data_dir, min_zoom=12))

        # Use make_tiles with new signature
        karttapullautin2tiles.make_tiles(
            in_dir=test_data_dir, out_dir=out_dir, tiles=tiles, proj="EPSG:25832", pattern="*depr*.pgw", max_zoom=14
        )

        # Check that some tiles were created
        assert out_dir.exists()
        zoom_dirs = list(out_dir.glob("*"))
        assert len(zoom_dirs) > 0

        # Check directory structure exists (z/x/y.png)
        for zoom_dir in zoom_dirs:
            if zoom_dir.is_dir():
                x_dirs = list(zoom_dir.glob("*"))
                if x_dirs:  # Only check if there are x directories
                    assert any(x_dir.is_dir() for x_dir in x_dirs)
                    for x_dir in x_dirs:
                        if x_dir.is_dir():
                            png_files = list(x_dir.glob("*.png"))
                            if png_files:  # Only check if there are PNG files
                                assert all(f.suffix == ".png" for f in png_files)


def test_make_tiles_empty_tiles_list():
    """Test make_tiles with empty tiles list"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        karttapullautin2tiles.make_tiles(
            in_dir=Path("/nonexistent"),  # Won't be accessed due to empty tiles
            out_dir=out_dir,
            tiles=[],
            max_zoom=14,
        )

        # Should complete without error
        assert out_dir.exists()


@pytest.mark.parametrize("max_zoom", [13, 15, 17])
def test_make_tiles_different_max_zoom(test_data_dir, max_zoom):
    """Test make_tiles with different max zoom levels"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        tiles = list(karttapullautin2tiles.list_tiles(test_data_dir, min_zoom=12))
        if not tiles:
            pytest.skip("No tiles found for testing")

        karttapullautin2tiles.make_tiles(
            in_dir=test_data_dir,
            out_dir=out_dir,
            tiles=tiles[:1],  # Use only first tile for faster testing
            max_zoom=max_zoom,
        )

        assert out_dir.exists()


# Tests for get_html_viewer function
def test_get_html_viewer_basic():
    """Test HTML viewer generation"""
    html = karttapullautin2tiles.get_html_viewer(
        lon_center=10.0, lat_center=50.0, default_zoom=12, min_zoom=8, max_zoom=16
    )

    assert isinstance(html, str)
    assert len(html) > 0
    assert "10.0" in html  # lon_center should be in HTML
    assert "50.0" in html  # lat_center should be in HTML
    assert "12" in html  # default_zoom should be in HTML


def test_get_html_viewer_parameters():
    """Test that HTML viewer contains all specified parameters"""
    lon_center, lat_center = -120.5, 35.2
    default_zoom, min_zoom, max_zoom = 10, 5, 18

    html = karttapullautin2tiles.get_html_viewer(
        lon_center=lon_center, lat_center=lat_center, default_zoom=default_zoom, min_zoom=min_zoom, max_zoom=max_zoom
    )

    assert str(lon_center) in html
    assert str(lat_center) in html
    assert str(default_zoom) in html
    assert str(min_zoom) in html
    assert str(max_zoom) in html


# Tests for module constants
def test_wgs84_crs_constant():
    """Test that WGS84 CRS constant is correct"""
    assert karttapullautin2tiles.WGS_84_CRS == "EPSG:4326"


def test_logger_exists():
    """Test that logger is properly configured"""
    assert hasattr(karttapullautin2tiles, "logger")
    assert karttapullautin2tiles.logger.name == "karttapullautin2tiles"
