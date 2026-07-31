import json
import tempfile
from pathlib import Path

import cyclopts
import pytest

from karttapullautin2tiles import cli


@pytest.fixture
def tile_list_file(test_data_dir):
    """A one-line tile list, as produced by `k2t list-tiles`"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tile_list = Path(tmp_dir) / "tiles.json"
        tile_list.write_text(json.dumps({"x": 2162, "y": 1432, "z": 12}) + "\n")
        yield tile_list


def test_make_tiles_rejects_webp_method_with_png(tile_list_file, test_data_dir):
    """--webp-method is only meaningful for webp output, so combining it with png is an error"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(cyclopts.ValidationError, match="only applies to --format webp"):
            cli.app(
                [
                    "make-tiles",
                    str(test_data_dir),
                    tmp_dir,
                    str(tile_list_file),
                    "--proj",
                    "EPSG:25832",
                    "--format",
                    "png",
                    "--webp-method",
                    "3",
                ],
                exit_on_error=False,
            )


@pytest.mark.parametrize("webp_method", [0, 2, 6])
def test_make_tiles_accepts_webp_method_with_webp(tile_list_file, test_data_dir, webp_method):
    """Any valid effort level is accepted with webp output, including 0"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        # The tile in the list does not overlap the test data, so nothing is written; this
        # exercises argument handling without paying for a full pyramid.
        cli.app(
            [
                "make-tiles",
                str(test_data_dir),
                str(out_dir),
                str(tile_list_file),
                "--proj",
                "EPSG:25832",
                "--webp-method",
                str(webp_method),
            ],
            exit_on_error=False,
            result_action="return_value",
        )
        assert out_dir.exists()


def test_make_tiles_viewer_uses_tile_extension(tile_list_file, test_data_dir):
    """The generated viewer must request the extension that was actually written"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        cli.make_tiles(
            in_dir=test_data_dir,
            out_dir=out_dir,
            tile_list=tile_list_file,
            proj="EPSG:25832",
            tile_format="png",
            max_zoom=12,
        )

        html = (out_dir / "viewer.html").read_text()
        assert "{z}/{x}/{y}.png" in html
        assert "maxZoom: 12" in html
