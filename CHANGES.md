# Changelog

## [Unreleased]

First release.

### Added

- `drizzle_operator`: the drizzle square-kernel operator of one exposure
  as a `scipy.sparse` CSR array, from its pixel map, `pixfrac`, input
  weights and weight scale.  Its steps, `corners`, `jacobian`,
  `overlap_triplets` and `build_operator`, are public so that the overlap
  geometry can be kept and re-weighted.
- `overlap_triplets` computes the overlap areas with drizzle's `boxer`
  and `sgarea`, ported to numba, in parallel over input rows; the results
  do not depend on the number of threads.
- `Coadd`: the weighted mean of exposures on one output grid, with any
  number of variance planes and drizzle's context image, summed in
  float64 and divided once at the end.
- Every input pixel with a valid pixel map is drizzled, so output pixels
  at the edges keep all their weight when the inputs extend past the
  output frame; drizzle skips input pixels centred outside the frame.
- Tests against drizzle with float64 output arrays: single-pixel
  operator columns, weights, science and variance for several `pixfrac`,
  rotations, scales, a mirrored input, NaN pixels of zero weight,
  `iscale`, weight scale, two exposures, context, and partial overlap.

[Unreleased]: https://github.com/mpi-astronomy/spizzle/commits/main
