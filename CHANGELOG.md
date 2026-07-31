# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html


## v0.2.0

### Features
* Tiles are now written as lossless WebP by default, which is ~25% smaller than PNG and decodes faster.
  Use `--format png` for consumers that cannot read WebP, and `--webp-method` to trade encoding time
  for file size.
* Downscale tiles with `average` instead of `lanczos` resampling. Lanczos rings around the sharp lines
  of a map, which added halos and roughly doubled the number of distinct colours in a tile. Combined
  with WebP this reduces the size of a tile directory by ~42% with no loss of quality.
* Fix that the viewer requested tiles beyond the generated maximum zoom level, resulting in blank tiles.
  It now upscales the highest generated zoom level instead.

### Chore
* Update to `cookiecutter-scverse` template v0.8.0
* Require Python >= 3.12 (following the template); support for Python 3.10 and 3.11 is dropped

## v0.1.3

* Fix that intersection of target tile with geopandas data frame had not been correctly calculated, leading to white wedges on the map
* Fix that min_zoom level has not been correctly respected

## v0.1.2

Remove default value for `--proj` parameter

## v0.1.1

Remove unnecessary dependencies

## v0.1.0

Initial release
