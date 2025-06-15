import tempfile
from pathlib import Path

import geopandas as gpd
import karttapullautin2tiles
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon


@pytest.fixture
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "data" / "karttapullautin_out"


@pytest.fixture
def sample_pgw_file(test_data_dir):
    """Path to a sample PGW file for testing"""
    return test_data_dir / "576_5265.laz_depr.pgw"


@pytest.fixture
def sample_geodataframe(test_data_dir):
    """Sample GeoDataFrame loaded from test data"""
    return karttapullautin2tiles.load_karttapullautin_dir(test_data_dir)


@pytest.fixture
def temp_image_files():
    """Create temporary image files for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test PGW file
        pgw_content = "1.0\n0\n0\n-1.0\n100.0\n200.0\n"
        pgw_file = tmp_path / "test.pgw"
        pgw_file.write_text(pgw_content)

        # Create test PNG file
        png_file = tmp_path / "test.png"
        test_img = Image.new("RGB", (10, 10), color="red")
        test_img.save(png_file)

        yield {"pgw_file": pgw_file, "png_file": png_file, "tmp_dir": tmp_path}


@pytest.fixture
def test_array():
    """Sample numpy array for testing array operations"""
    return np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)


@pytest.fixture
def sample_polygon():
    """Sample polygon for testing geometry operations"""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.fixture
def empty_geodataframe():
    """Empty GeoDataFrame for testing edge cases"""
    return gpd.GeoDataFrame(columns=["id", "pgw_file", "img_file", "geometry"], crs="EPSG:25832")
