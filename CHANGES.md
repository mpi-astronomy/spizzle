# Changelog

## [Unreleased]

First release.

### Added

- `drizzle_operator`: the drizzle square-kernel operator of one exposure,
  from its pixel map, `pixfrac`, input weights and weight scale.  Its
  steps, `corners`, `jacobian`, `overlap_triplets` and `build_operator`,
  are public so that the overlap geometry can be kept and re-weighted.
- `Operator`: the operator of one exposure as a `scipy.sparse` CSR array
  on the exposure's bounding box of the output grid, with the box, so
  that building and applying it scale with the exposure rather than with
  the output grid; `Operator.full()` gives the CSR array on the whole
  output grid.
- `corners`, `jacobian`, `overlap_triplets`, `build_operator` and
  `Coadd.add` run in parallel on numba's threads; the results do not
  depend on the number of threads.  `overlap_triplets` computes the
  overlap areas with drizzle's `boxer` and `sgarea`, ported to numba.
- `pixel_map`: the pixel map of an exposure from its WCS and the output
  WCS, as drizzle's `calc_pixmap` makes it.  By default the WCS are
  evaluated on every 16th pixel and interpolated with bicubic splines,
  which for NIRCam is 11 times faster and differs from the full
  evaluation by at most 2e-8 pixel; `stepsize=1` evaluates every pixel,
  on several threads.
- `Coadd`: the weighted mean of exposures on one output grid, with any
  number of variance planes and drizzle's context image, summed in
  float64 and divided once at the end.
- Every input pixel with a valid pixel map is drizzled, so output pixels
  at the edges keep all their weight when the inputs extend past the
  output frame; drizzle skips input pixels centered outside the frame.
- `benchmarks/bench_spizzle.py`: timings and peak memory against
  drizzle's C square kernel.
- `benchmarks/compare_jwst.py`: drizzles JWST `cal` or `crf` files with
  drizzle and spizzle, compares the outputs and times both.
- Tests against drizzle with float64 output arrays: single-pixel
  operator columns, weights, science and variance for several `pixfrac`,
  rotations, scales, a mirrored input, NaN pixels of zero weight,
  `iscale`, weight scale, two exposures, context, and partial overlap.

[Unreleased]: https://github.com/mpi-astronomy/spizzle/commits/main
