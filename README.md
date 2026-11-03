# spizzle: drizzle as a sparse linear operator

spizzle resamples images with the square kernel of
[drizzle](https://github.com/spacetelescope/drizzle), written out as an
explicit sparse matrix `W`.  It builds `W` once per exposure and applies
it to the science data, the weights and every variance plane, in float64
and in parallel.  With `W` in hand it also gives the noise covariance that
drizzling introduces, so that mosaics come with correct uncertainties for
sums over areas typical in aperture photometry.


# What is here

| | |
| --- | --- |
| `pixel_map` | the pixel map of an exposure onto an output grid, from the two WCS; fast, by splines through a coarse grid |
| `drizzle_operator` | the square-kernel operator `W` of one exposure, an `Operator` on the exposure's region of the output grid |
| `Coadd` | the weighted mean of exposures: science, weights, variance planes and, if asked for, context, summed in float64 |
| `Covariance` | the covariance between neighboring output pixels, the noise factor, and a correlation file |
| `NoiseFactor` | the noise factor alone, in a second pass over the exposures, with little memory |
| `blot` | a mosaic interpolated back onto an exposure's pixels, as drizzle's bilinear `blot_image` |
| `StcalVariance` | the variance planes and ERR as the JWST pipeline computes them, to compare with it |

`corners`, `jacobian`, `overlap_triplets` and `build_operator` are the
steps of `drizzle_operator`, to keep the geometry and re-weight it.


# Installation

    pip install spizzle

Install extras `pip install "spizzle[jwst]"` adds jwst for comparing
spizzle with the JWST pipeline (`benchmarks/compare_stcal.py`).


# Getting started

```python
import numpy as np
from spizzle import Coadd, NoiseFactor, drizzle_operator, pixel_map

# exposures: (data, var, weights, wcs) of each exposure; wcs_out, out_shape: the output grid
coadd = Coadd(out_shape, n_var=1)
ops = []
for data, var, weights, wcs in exposures:
    pixmap = pixel_map(wcs, wcs_out, shape=data.shape)
    op = drizzle_operator(pixmap, out_shape, pixfrac=0.8, weights=weights)
    coadd.add(op, data, variances=[var])
    ops.append(op)

image, weight, variance = coadd.out_img, coadd.out_wht, coadd.out_var[0]

# the noise factor, in a second pass with the weight sums of the first
# (keep the operators, as here, or build them again)
nf = NoiseFactor(coadd.out_wht)
for op, (data, var, weights, wcs) in zip(ops, exposures):
    nf.add(op, var)
error = np.sqrt(variance * nf.noise_factor)   # for sums over areas
```

`sqrt(variance)` is the uncertainty of a single pixel; drizzling correlates
neighboring pixels, so the uncertainty of a sum over an area is larger by
the square root of the noise factor, about 1.5 for NIRCam at its native
scale.  To compare an exposure with a mosaic, as outlier detection does,
`blot(median, pixmap)` interpolates the mosaic back onto the exposure's
pixels, with the exposure's pixel map onto the mosaic.

The [documentation](https://mpi-astronomy.github.io/spizzle/) describes the pixel maps, the
operator, the covariance and blot in more detail, and how the results
compare with drizzle and with the JWST pipeline.


# Speed

One NIRCam exposure (2048x2048, three variance planes) onto a mosaic of
eight, on an Apple M3 Pro with 11 threads (`benchmarks/speed_table.py`):

| step | drizzle | spizzle | faster |
| --- | ---: | ---: | ---: |
| pixel map from the WCS | 0.77 s | 0.020 s | 38x |
| drizzle the science data and 3 variance planes | 0.51 s | 0.24 s | 2x |
| blot a median and its error | 0.039 s | 0.0058 s | 7x |
| jwst `ResampleImage` of the eight exposures | 20 s | 4.8 s | 4x |

The results agree with drizzle computing in float64 to about 1e-15, and
with the JWST pipeline to float32 round-off.  In the last row spizzle
computes the variance as the pipeline does (`StcalVariance`), with the
data already in memory.


# How it differs from drizzle

 - Square kernel only.
 - Every input pixel with a valid pixel map is drizzled; drizzle skips
   input pixels whose centers fall outside the output frame, so the outer
   output pixels get less weight from drizzle than from spizzle.
 - `blot` accepts positions over the image's extent, `[-0.5, n - 0.5)`;
   drizzle accepts `[0, n)`, half a pixel off.
 - Fill values are numbers, NaN by default.


# Development

The tests compare spizzle with drizzle, which must accept float64 output
arrays.  Until that is in a drizzle release, install it from the branch:

    pip install -e ".[test]"
    pip install "drizzle @ git+https://github.com/jdavies-st/drizzle.git@float64-phase2-c-accepts-float64"
    pytest tests

`benchmarks/` has the timings and the comparisons with drizzle, the JWST
pipeline and noise realizations.


# Citing spizzle

The citation information is in `CITATION.cff`.


# License

BSD 3-clause, see `LICENSE`.  The overlap geometry in `spizzle.geometry`
and the bilinear interpolation of `blot` are ported from drizzle,
Copyright (C) 2011,2014 Association of Universities for Research in
Astronomy (AURA), under the BSD 3-clause license in
`licenses/DRIZZLE_LICENSE.rst`.
