# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`karttapullautin2tiles` (CLI: `k2t`) converts [karttapullautin](https://github.com/karttapullautin/karttapullautin)
output — one georeferenced PNG per LIDAR tile, each with a sidecar `.pgw` world file — into an
XYZ (`z/x/y.png`) web-mercator tile directory usable by web map viewers.

The CLI is deliberately split into `list-tiles` (enumerate min-zoom tiles covering the input) and
`make-tiles` (render one chunk of that list), so work can be parallelized with GNU parallel or
across machines. See README.md for user-facing usage.

## Development environment

Python 3.12–3.14 (`requires-python = ">=3.12"`). Dev/test/doc dependencies live in
`[dependency-groups]`, not in extras, so they are requested with `--group`:

```bash
uv sync --group test        # dev group is included by default
uv run pytest               # 39 tests, ~25 s
```

The geospatial stack only gained cp314 wheels recently (`pyproj>=3.7.2`, `rasterio>=1.5.0`); on
older pins uv tries to build them from source and fails with `proj executable not found`, since
there is no system PROJ/GDAL here. If that happens, `uv lock --upgrade` rather than downgrading
Python.

## Tests

CI (`.github/workflows/test.yaml`) reads its matrix from `tool.hatch.envs.hatch-test.matrix` in
`pyproject.toml` and runs the tests through hatch — `pyproject.toml` is the single source of truth,
`uv.lock` is not used by CI.

Hatch creates its envs with uv and inherits the container's `UV_PYTHON=3.14`, which makes it install
the dependencies outside the env it just created (`pytest: not found`, or
`No virtual environment found for Python 3.14`). Unset it per command — and if an env was already
created that way, `hatch env remove <env>` before retrying, because the broken env is cached:

```bash
env -u UV_PYTHON uvx hatch test -py 3.12              # one interpreter
env -u UV_PYTHON uvx hatch test                       # all matrix envs (3.12 + 3.14 stable, 3.14 pre)
env -u UV_PYTHON uvx hatch test -py 3.12 tests/test_make_tiles.py::test_make_tiles_basic  # single test
env -u UV_PYTHON uvx hatch run hatch-test.py3.14-stable:run-cov -n auto   # exactly what CI runs
```

pytest config lives in `[tool.pytest]` (pytest 9's native table, not `[tool.pytest.ini_options]`).

`tests/test_make_tiles.py` covers private helpers directly (`_load_img`, `_subset_array`,
`_get_tile_bb_polygon`, `_get_tiles_center`) as well as the public functions, so renaming a helper
breaks tests. Fixtures live in `tests/conftest.py`; real fixture data (EPSG:25832) is in
`tests/data/karttapullautin_out/`. `testdata/` at the repo root is untracked scratch data with a
different naming scheme (no `.laz` in the stem) — not used by the suite.

## Lint / format

Everything runs through pre-commit (hooks are installed via `prek` in this container):

```bash
prek run --all-files          # or: pre-commit run --all-files
```

Ruff: line length 120, numpy docstring convention, pydocstyle enabled on `src/` (tests exempt).
`biome-format` formats JSON/JSONC, `pyproject-fmt` formats `pyproject.toml` — expect those files to
be rewritten on commit. `zizmor` audits the GitHub workflows, which is why every action is pinned to
a commit hash (dependabot bumps them); keep the hash + `# vX.Y.Z` comment form when editing them.

## Architecture

`src/karttapullautin2tiles/__init__.py` holds the entire library; `cli.py` is a thin
[cyclopts](https://cyclopts.readthedocs.io/) wrapper whose docstrings *are* the `--help` output
(README's command reference is pasted from it, so update both together).

Pipeline, in call order:

1. `_load_img(pgw_file)` — parses the 6-line `.pgw` world file (pixel size x, 2 rotation terms,
   pixel size y (negative), top-left x, top-left y) plus the PNG dimensions from `imagesize`, and
   returns the image footprint as a shapely `Polygon`. The PNG is located as the `.png` sibling of
   the pgw file (`Path.with_suffix`), so `576_5265.laz_depr.pgw` → `576_5265.laz_depr.png`.
2. `load_karttapullautin_dir(dir, proj=, pattern=)` — one row per footprint in a `GeoDataFrame`
   whose CRS is the caller-supplied `--proj` (the PNGs carry no georeferencing themselves).
   Returns an empty frame with the same columns when nothing matches.
3. `list_tiles(...)` — reprojects `gpdf.total_bounds` to WGS84 and yields `mercantile.tiles(...)` at
   `min_zoom`. `cli.list_tiles` serializes each tile as one JSON object per line
   (`{"x": ..., "y": ..., "z": ...}`) on stdout; `cli.make_tiles` reads that format from a file or stdin.
4. `make_tiles(in_dir, out_dir, tiles, ...)` — for each min-zoom "parent" tile:
   `_get_tile_bb_polygon` converts the tile bbox into the source CRS, selects intersecting
   footprints, `rasterio.merge.merge` stitches those PNGs into a single in-memory array, then every
   child tile from `parent.z` through `max_zoom` is rendered from that array.
5. `extract_and_transform_tile(...)` — `_subset_array` crops the merged array to the tile bbox plus
   10% padding (purely a `reproject` speed optimization), then `rasterio.warp.reproject` warps it to
   the tile's WGS84 bounds at 256×256 with lanczos resampling, returning a PIL image saved to
   `out_dir/z/x/y.png`.
6. `get_html_viewer(...)` — string-substitutes `{{lon_center}}`, `{{lat_center}}`,
   `{{default_zoom}}`, `{{min_zoom}}`, `{{max_zoom}}` into
   `src/karttapullautin2tiles/assets/local_tiles_viewer.html` (loaded via `importlib.resources`, so
   the assets dir must ship in the wheel) and is written as `viewer.html` in the output dir.

### Conventions that are load-bearing

- **Nodata is white (`255`), not transparent**, throughout `merge`/`reproject`/the pre-filled
  destination array. A tile with no coverage is a blank white PNG: `extract_and_transform_tile`
  catches the `ValueError` that `_subset_array` raises for non-overlapping tiles and returns the
  pre-filled array unchanged. Callers therefore never see "missing tile" as an error.
- **Memory is bounded by the min-zoom tile**, since every PNG intersecting one parent tile is merged
  at once. This is why `list-tiles --zoom` exists; raising it is the fix for OOM.
- **Default `pattern="*depr*.pgw"`** picks karttapullautin's depression-shaded rendering. Changing it
  changes which PNG variant ends up in the tiles.
- `__version__` comes from installed package metadata (`importlib.metadata.version`) with the version
  itself derived from git tags by `hatch-vcs`. Importing from a checkout that was never installed
  raises `PackageNotFoundError`.
- `cli.make_tiles` passes `max_zoom=-max_zoom` to `get_html_viewer` (`cli.py:108`). The template
  currently hardcodes `maxZoom: 18` and ignores the value, so this is inert — fix it before making
  the template use `{{max_zoom}}`.

## Repo housekeeping

- Templated from [cookiecutter-scverse](https://github.com/scverse/cookiecutter-scverse) (currently
  pinned to `v0.8.0` in `.cruft.json`) and synced by a bot that opens a PR per template release.
  A pre-commit hook blocks committing `.rej` files left behind by such a merge — resolve those
  conflicts by hand. `uvx cruft diff` lists remaining drift from the template; the only files
  expected to differ are `pyproject.toml`, `README.md`, `CHANGELOG.md`, `LICENSE`, `.gitignore` and
  `.github/ISSUE_TEMPLATE/config.yml` (deliberately dropped the scverse forum link — this is not an
  scverse package). Files this repo intentionally does not want from the template go into
  `[tool.cruft] skip` in `pyproject.toml`.
- The template's `.cruft.json` as written by the sync bot has `"template": "/tmp/tmp…"` — a temp
  clone path. Restore it to `https://github.com/scverse/cookiecutter-scverse` when merging.
- Add a bullet to `CHANGELOG.md` for user-visible changes.
- Releasing = publish a GitHub Release; `.github/workflows/release.yaml` builds with `uv build` and
  uploads to PyPI via trusted publishing. The tag drives the version.
- `docs/` and `.readthedocs.yaml` were deliberately removed (`d125af0`) and are listed in
  `[tool.cruft] skip` so template syncs stop reintroducing them. The template's `doc` dependency
  group and `envs.docs` scripts are still there, so `hatch run docs:build` will fail — that is
  expected, not a regression.
