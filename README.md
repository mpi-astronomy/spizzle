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

 - `drizzle_operator`: the operator of one exposure, a `scipy.sparse`
   CSR array of shape (output pixels, input pixels)
 - `corners`, `jacobian`, `overlap_triplets`, `build_operator`: the steps
   of `drizzle_operator`, to keep the overlap geometry and re-weight it
   without recomputing it
 - `Coadd`: the weighted mean of exposures on one output grid, with
   variance planes and drizzle's context image


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
   input pixels whose centres fall outside the output frame, so when the
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
`Coadd.add` about 0.02 s per plane.  The science data with three variance
planes take 0.32 s, against 0.72 s for drizzle with the variance planes in
the same call; each further pass over the same geometry, such as a noise
realisation, costs about 0.02 s, against 0.56 s for a drizzle call.  On a
single thread spizzle is slower than drizzle.  The peak memory is higher:
3.0 GiB for the exposure with three variance planes, against 1.25 GiB.


# Installation

    pip install git+https://github.com/mpi-astronomy/spizzle


# Usage

```python
from spizzle import Coadd, drizzle_operator

coadd = Coadd(out_shape, n_var=1)
for data, var, weights, pixmap in exposures:
    W = drizzle_operator(pixmap, out_shape, pixfrac=0.8, weights=weights)
    coadd.add(W, data, variances=[var])

image = coadd.out_img
weight = coadd.out_wht
variance = coadd.out_var[0]
context = coadd.out_ctx
```

The pixel map holds the output `x` and `y` of every input pixel centre,
shape `(ny, nx, 2)`, NaN where an input pixel has no mapping.  From two
WCS objects with the common astropy interface (gwcs or `astropy.wcs`):

```python
import numpy as np

y, x = np.indices(data.shape)
xo, yo = wcs_out.world_to_pixel(wcs_in.pixel_to_world(x, y))
pixmap = np.dstack([xo, yo])
```

or `drizzle.utils.calc_pixmap(wcs_in, wcs_out)`.

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
W = build_operator(triplets, jaco, out_shape, weights=weights)
```

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
