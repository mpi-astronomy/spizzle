# spizzle: drizzle as a sparse linear operator

spizzle resamples images with the square kernel of
[drizzle](https://github.com/spacetelescope/drizzle), written out as an
explicit sparse matrix.  Drizzling an exposure is a linear map `W @ data`,
and `W` depends on the pixel map, `pixfrac` and the input weights, not on
the data.  spizzle builds `W` once per exposure and applies it to the
science data, the weights and every variance plane, both in parallel.
The sums are kept in float64 and divided once at the end, instead of
updating a float32 weighted mean for every contribution.  `W` itself is
there for noise propagation: with independent input noise the covariance
of the weighted sums `W @ data` is `W @ diag(var) @ W.T`, and the output
image is those sums divided by the output weights.

 - `drizzle_operator`: the operator of one exposure, an `Operator`: a
   `scipy.sparse` CSR array of shape (box pixels, input pixels) on the
   exposure's bounding box of the output grid, the box, and `full()` for
   the CSR array on the whole output grid
 - `corners`, `jacobian`, `overlap_triplets`, `build_operator`: the steps
   of `drizzle_operator`, to keep the overlap geometry and re-weight it
   without recomputing it
 - `Coadd`: the weighted mean of exposures on one output grid, with
   variance planes and drizzle's context image
 - `pixel_map`: the pixel map of an exposure from its WCS and the output
   WCS, by default interpolated from the WCS on a coarse grid
 - `Covariance`: the covariance between nearby pixels of the co-added
   image, for independent input noise


# How it compares with drizzle

The geometry is drizzle's: the interpolation of the pixel corners from
the pixel map, the Jacobian, and drizzle's `boxer` and `sgarea` for the
overlap areas are ported with the same order of floating point
operations.  Against drizzle built without fused multiply-adds
(`CFLAGS=-ffp-contract=off`) the output weights agree bit for bit; with a
default build, which lets the compiler fuse them, to about 1e-15.  The
science and variance outputs agree with drizzle's weighted mean computed
in float64 to float64 round-off.

The differences:

 - Square kernel only; `pixfrac` must be larger than 0.001, below which
   drizzle switches to the point kernel.
 - Every input pixel with a valid pixel map is drizzled.  drizzle skips
   input pixels whose centers fall outside the output frame, so when the
   inputs extend past the output, the outer two rings of output pixels
   get less weight from drizzle than from spizzle.  Away from those
   edges the results are the same.
 - As in drizzle, an input pixel with NaN data and non-zero weight makes
   the output pixels it lands on NaN.  Give such pixels zero weight, as
   the jwst pipeline does; pixels with zero weight are left out of `W`
   and contribute nothing.


# Speed

`benchmarks/bench_spizzle.py` times spizzle against drizzle's C kernel.
On an Apple M3 Pro (11 threads), for a 2048x2048 exposure drizzled at 2x
oversampling with `pixfrac=0.8`, building `W` takes 0.26 s and
`Coadd.add` about 0.012 s per plane.  The science data with three variance
planes take 0.30 s, against 0.72 s for drizzle with the variance planes in
the same call; each further pass over the same geometry, such as a noise
realization, costs about 0.015 s, against 0.56 s for a drizzle call.  On a
single thread spizzle is slower than drizzle.  The peak memory is higher:
3.0 GiB for the exposure with three variance planes, against 1.25 GiB.

`benchmarks/compare_jwst.py` drizzles a set of JWST exposures with both
packages and compares the results.  For eight NIRCam long-wavelength
exposures on a common mosaic frame the two agree to better than 2e-12
(most pixels better than 1e-15) against drizzle computing in float64, and
spizzle takes 0.26 s per exposure against drizzle's 0.42 s.  Evaluating
the WCS for every pixel with `drizzle.utils.calc_pixmap` takes 0.84 s;
spizzle's `pixel_map` takes 0.022 s and differs from it by at most 2e-8
pixel.


# Installation

    pip install git+https://github.com/mpi-astronomy/spizzle


# Usage

```python
from spizzle import Coadd, drizzle_operator, pixel_map

coadd = Coadd(out_shape, n_var=1)
for data, var, weights, wcs in exposures:
    pixmap = pixel_map(wcs, wcs_out, shape=data.shape)
    op = drizzle_operator(pixmap, out_shape, pixfrac=0.8, weights=weights)
    coadd.add(op, data, variances=[var])

image = coadd.out_img
weight = coadd.out_wht
variance = coadd.out_var[0]
context = coadd.out_ctx
```

The pixel map holds the output `x` and `y` of every input pixel center,
shape `(ny, nx, 2)`, NaN where an input pixel has no mapping.
`pixel_map` makes it from two WCS objects with the shared astropy
interface (gwcs or `astropy.wcs`) that use the same celestial frame, as
drizzle's `calc_pixmap` does.  Evaluating a WCS for every pixel is slow,
so by default `pixel_map` evaluates it on every 16th pixel and
interpolates with bicubic splines, which assumes a smooth distortion;
`stepsize=1` evaluates every pixel, on several threads.  Any pixel map
works, such as one from `drizzle.utils.calc_pixmap(wcs_in, wcs_out)`.

An exposure usually covers a small part of the output grid, which can be
a large mosaic, so `drizzle_operator` keeps `W` to the exposure's
bounding box: `op.matrix @ data.ravel()` gives the weighted sums on the
pixels of `op.box`, a pair of slices, so that `image[op.box]` is the
exposure's region of an output image.  Building and applying the
operator then scale with the exposure, not with the output grid.
`op.full()` gives `W` on the whole output grid, and `Coadd.add` takes
either.

`Coadd.add` takes drizzle's `iscale` for the data (its square for the
variances), and `drizzle_operator` takes drizzle's `wht_scale` as
`weight_scale`.  Unit conversions, such as dividing data in counts by the
exposure time, go in `iscale`.

To re-weight an exposure without recomputing its geometry:

```python
from spizzle import build_operator, corners, jacobian, overlap_triplets

c = corners(pixmap, pixfrac=0.8)
triplets = overlap_triplets(c, out_shape)
jaco = jacobian(c)
op = build_operator(triplets, jaco, out_shape, weights=weights)
```

Drizzling correlates the noise of neighboring output pixels.
`Covariance` adds up `W diag(var) W.T` for the same operators, one
variance image per exposure, and gives the covariance of the image
between each pixel and its neighbors:

```python
import numpy as np
from spizzle import Covariance

cov = Covariance(out_shape)
for data, var, weights, wcs in exposures:
    ...
    coadd.add(op, data, variances=[var])
    cov.add(op, var)

err = np.sqrt(coadd.out_var[0] * cov.noise_factor)  # for sums over areas
cov.write_correlation("mosaic_corr.fits")            # optional, separate file
```

`cov.noise_factor` is, for each pixel, its covariance with all pixels
over its variance: the variance of a sum over an area much larger than
an input pixel's footprint is the sum of its pixels' variances times this
factor.  `sqrt(var * noise_factor)` is therefore the uncertainty for
aperture photometry and other sums over areas, while `sqrt(var)` stays
that of a single pixel.  For eight NIRCam exposures at native scale the
factor has a median of 2.33 and ranges from 1.65 to 3.38 (5 to 95%), where
the usual single correction (Casertano et al. 2000) is 2.25 everywhere.

`cov.covariance(k)` and `cov.correlation(k)` give the covariance and the
correlation of each pixel with its neighbor at `cov.offsets[k]`, for half
of the offsets up to the radius, and `write_correlation` writes all the
correlation images to a FITS file, one at a time.  Pixels farther apart
than an input pixel's footprint share no input pixel and are
uncorrelated; each `Operator` records that distance as `reach`, and by
default the radius is the reach of the first exposure added.  The sums
take `2 r (r + 1) + 1` float32 images the size of the output: 13 for
radius 2 (native scale, pixfrac 1), 1.4 GiB for the NIRCam mosaic above.
Checked against noise realizations, and for evenly dithered exposures it
gives Casertano's correction to about 1% (`benchmarks/check_covariance.py`).

The geometry, `build_operator` and `Coadd.add` run on numba's threads;
set their number with `numba.set_num_threads` or the `NUMBA_NUM_THREADS`
environment variable.  The results do not depend on the number of
threads.


# Citing spizzle

The citation information is in `CITATION.cff`.


# Development

The tests compare spizzle with drizzle, which must accept float64 output
arrays.  Until that is in a drizzle release, install it from the branch:

    pip install -e ".[test]"
    pip install "drizzle @ git+https://github.com/jdavies-st/drizzle.git@float64-phase2-c-accepts-float64"
    pytest tests


# License

BSD 3-clause, see `LICENSE`.  The overlap geometry in `spizzle.geometry`
is ported from drizzle, Copyright (C) 2011,2014 Association of
Universities for Research in Astronomy (AURA), under the BSD 3-clause
license in `licenses/DRIZZLE_LICENSE.rst`.
