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
  which for NIRCam is about 40 times faster and differs from the full
  evaluation by at most 2e-8 pixel; `stepsize=1` evaluates every pixel,
  on several threads.
- `Covariance`: the covariance between nearby pixels of the co-added
  image, for independent input noise, as one image per pixel offset up
  to a radius, by default the reach of the operators, and summed in
  float32 unless asked for float64; in float64 its zero offset equals
  `Coadd.out_var` bit for bit.  `noise_factor` gives each pixel's
  covariance with all pixels over its variance, the per-pixel correction
  for the noise of sums over areas; `correlation` the correlation with a
  neighbor; and `write_correlation` writes all correlation images to a
  separate FITS file (needs astropy).
- `blot`: an image interpolated at the positions of a pixel map, as
  drizzle's `blot_image` with bilinear interpolation, which stcal's
  `gwcs_blot` uses for outlier detection.  It matches drizzle bit for
  bit, NaN positions included, and runs in parallel on numba's threads.
  It accepts positions over the image's extent, `[-0.5, n - 0.5)`, where
  drizzle accepts `[0, n)`, and holds the edge values beyond the
  outermost pixel centers.
- `Operator.reach`: the largest distance in y or x between two output
  pixels that share an input pixel, beyond which drizzled noise is
  uncorrelated.
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
- `benchmarks/check_covariance.py`: checks `Covariance` against noise
  realizations and the noise correction of Casertano et al. (2000).
- Tests against drizzle with float64 output arrays: single-pixel
  operator columns, weights, science and variance for several `pixfrac`,
  rotations, scales, a mirrored input, NaN pixels of zero weight,
  `iscale`, weight scale, two exposures, context, and partial overlap.

[Unreleased]: https://github.com/mpi-astronomy/spizzle/commits/main
