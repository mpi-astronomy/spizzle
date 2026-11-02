spizzle
=======

Drizzle's square kernel written out as an explicit sparse linear operator.
Drizzling an exposure is the linear map ``W @ data``, and ``W`` depends on
the pixel map, ``pixfrac`` and the input weights, not on the data.  spizzle
builds ``W`` once per exposure and applies it to the science data, the
weights and every variance plane, both in parallel, summing in float64 and
dividing once at the end.  ``W`` is also what noise propagation needs:
with independent input noise the covariance of the weighted sums is
``W @ diag(var) @ W.T``.

Usage
-----

.. code-block:: python

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

Pixel maps
----------

The pixel map holds the output ``x`` and ``y`` of every input pixel
center, shape ``(ny, nx, 2)``, NaN where an input pixel has no mapping.
`pixel_map` makes it from two WCS objects with the shared astropy
interface (gwcs or ``astropy.wcs``) that use the same celestial frame, as
drizzle's ``calc_pixmap`` does.  Evaluating a WCS for every pixel is slow,
so by default `pixel_map` evaluates it on every 16th pixel and
interpolates with bicubic splines, which assumes a smooth distortion: for
NIRCam it is about 38 times faster than ``calc_pixmap`` and differs from
it by at most 2e-8 pixel.  ``stepsize=1`` evaluates every pixel, on
several threads.  Any pixel map works, such as one from
``drizzle.utils.calc_pixmap(wcs_in, wcs_out)``.

The operator
------------

An exposure usually covers a small part of the output grid, which can be
a large mosaic, so `drizzle_operator` keeps ``W`` to the exposure's
bounding box: ``op.matrix @ data.ravel()`` gives the weighted sums on the
pixels of ``op.box``, a pair of slices, so that ``image[op.box]`` is the
exposure's region of an output image.  Building and applying the operator
then scale with the exposure, not with the output grid.  ``op.full()``
gives ``W`` on the whole output grid, and `Coadd.add` takes either.

`Coadd.add` takes drizzle's ``iscale`` for the data (its square for the
variances), and `drizzle_operator` takes drizzle's ``wht_scale`` as
``weight_scale``.  Unit conversions, such as dividing data in counts by
the exposure time, go in ``iscale``.

``drizzle_operator(..., region=(slice(y0, y1), slice(x0, x1)))`` drizzles
only the input pixels in that region, as drizzle's ``xmin``, ``xmax``,
``ymin`` and ``ymax`` do (theirs are inclusive, so ``y1 = ymax + 1``), and
skips the geometry of the others.  To leave out any other pixels, give
them zero weight: entries of zero weight are left out of the operator, so
their data are never used, NaN or not.

To re-weight an exposure without recomputing its geometry:

.. code-block:: python

    from spizzle import build_operator, corners, jacobian, overlap_triplets

    c = corners(pixmap, pixfrac=0.8)
    triplets = overlap_triplets(c, out_shape)
    jaco = jacobian(c)
    op = build_operator(triplets, jaco, out_shape, weights=weights)

Noise covariance
----------------

Drizzling correlates the noise of neighboring output pixels.
`Covariance` adds up ``W diag(var) W.T`` for the same operators as
`Coadd`, one variance image per exposure, and gives the covariance of the
image between each pixel and its neighbors:

.. code-block:: python

    import numpy as np
    from spizzle import Covariance

    cov = Covariance(out_shape)
    for data, var, weights, wcs in exposures:
        ...
        coadd.add(op, data, variances=[var])
        cov.add(op, var)

    err = np.sqrt(coadd.out_var[0] * cov.noise_factor)  # for sums over areas
    cov.write_correlation("mosaic_corr.fits")            # optional, separate file

``cov.noise_factor`` is, for each pixel, its covariance with all pixels
over its variance: the variance of a sum over an area much larger than an
input pixel's footprint is the sum of its pixels' variances times this
factor.  ``sqrt(var * noise_factor)`` is therefore the uncertainty for
aperture photometry and other sums over areas, while ``sqrt(var)`` stays
that of a single pixel.  For eight NIRCam exposures at native scale the
factor has a median of 2.33 and ranges from 1.65 to 3.38 (5 to 95%), where
the usual single correction (Casertano et al. 2000) is 2.25 everywhere.

``cov.covariance(k)`` and ``cov.correlation(k)`` give the covariance and
the correlation of each pixel with its neighbor at ``cov.offsets[k]``, for
half of the offsets up to the radius, and `Covariance.write_correlation`
writes all the correlation images to a FITS file, one at a time (needs
astropy).  Pixels farther apart than an input pixel's footprint share no
input pixel and are uncorrelated; each `Operator` records that distance as
``reach``, and by default the radius is the reach of the first exposure
added.  The sums take ``2 r (r + 1) + 1`` float32 images the size of the
output: 13 for radius 2 (native scale, pixfrac 1), 1.4 GiB for the NIRCam
mosaic of eight exposures.  Checked against noise realizations, and for
evenly dithered exposures it gives Casertano's correction to about 1%
(``benchmarks/check_covariance.py``).

When only the noise factor is needed, `NoiseFactor` gives it from two
sums instead of one per offset, in a second pass over the exposures once
`Coadd` has their weight sums:

.. code-block:: python

    from spizzle import NoiseFactor

    nf = NoiseFactor(coadd.out_wht)   # after every exposure is in coadd
    for data, var, weights, wcs in exposures:
        ...                           # the same operators, kept or built again
        nf.add(op, var)

    err = np.sqrt(coadd.out_var[0] * nf.noise_factor)

Each exposure adds ``W @ (var * (W.T @ (1 / wsum)))``, the covariance of
each pixel with every pixel over that pixel's weight sum, and
``(W * W) @ var``, so every offset counts and no radius is needed.  The
sums take 16 bytes per output pixel, against ``(2 r (r + 1) + 1) * 4 + 8``
for `Covariance`: 60 bytes at radius 2, 108 at radius 3.  For the NIRCam
mosaic that is 0.36 GiB against 1.37 GiB, for the same factor to 1.4e-7;
the second pass takes about as long as the first, unless the operators are
kept.

Blot
----

Outlier detection compares each exposure with the median of the resampled
exposures, interpolated back onto the exposure's pixels.  `blot` does that
interpolation, bilinear as drizzle's ``blot_image`` with
``interp="linear"`` (as stcal's ``gwcs_blot`` uses it).  Its pixel map is
the exposure's pixel map onto the grid of the median, the same one that
resampled it:

.. code-block:: python

    from spizzle import blot, pixel_map

    pixmap = pixel_map(wcs, wcs_median, shape=data.shape)
    blotted = blot(median, pixmap)            # NaN where nothing maps
    blotted_err = blot(median_err, pixmap)

It matches drizzle bit for bit except within half a pixel of the image's
edges.  There drizzle accepts positions in ``[0, n)``, half a pixel off the
image's extent ``[-0.5, n - 0.5)``; `blot` accepts the extent, holding the
edge values beyond the outermost pixel centers.  For a NIRCam exposure and
a median the size of the native mosaic, blotting the median and its error
takes 0.029 s, against 1.6 s for two calls of stcal's ``gwcs_blot``, each
of which evaluates the WCS for every pixel.  Most of the gain is the pixel
map, computed once from the splines; `blot` itself is about 7 times faster
than drizzle's.

The JWST pipeline's variance
----------------------------

The JWST pipeline does not propagate the variance through the drizzle
weights: stcal resamples the square root of each variance plane and
squares it, then combines the exposures with read-noise or exposure-time
weights.  `StcalVariance` computes the variance planes and ERR the same
way from spizzle's operators; the exact variance is `Coadd.out_var`.

``benchmarks/compare_stcal.py`` (with the ``jwst`` extra) runs jwst's own
``ResampleImage`` on JWST ``cal`` files and spizzle on the same exposures,
with stcal's weights and input ranges and jwst's output frame.  For eight
NIRCam exposures every plane agrees with jwst's to float32 round-off: SCI,
WHT, VAR_RNOISE, VAR_FLAT, VAR_POISSON and ERR to a median of 3-7e-8
relative, and CON exactly.

How it compares with drizzle
----------------------------

The geometry is drizzle's: the interpolation of the pixel corners from the
pixel map, the Jacobian, and drizzle's ``boxer`` and ``sgarea`` for the
overlap areas are ported with the same order of floating point operations.
Against drizzle built without fused multiply-adds
(``CFLAGS=-ffp-contract=off``) the output weights agree bit for bit; with a
default build, which lets the compiler fuse them, to about 1e-15.  The
science and variance outputs agree with drizzle's weighted mean computed
in float64 to float64 round-off: for eight NIRCam exposures on a common
mosaic frame to at most 2e-15 (``benchmarks/compare_jwst.py``).

The differences:

- Square kernel only; ``pixfrac`` must be larger than 0.001, below which
  drizzle switches to the point kernel.
- Every input pixel with a valid pixel map is drizzled.  drizzle skips
  input pixels whose centers fall outside the output frame, so when the
  inputs extend past the output, the outer two rings of output pixels get
  less weight from drizzle than from spizzle.  Away from those edges the
  results are the same.
- As in drizzle, an input pixel with NaN data and non-zero weight makes
  the output pixels it lands on NaN.  Give such pixels zero weight, as the
  jwst pipeline does; pixels with zero weight are left out of ``W`` and
  contribute nothing.
- `blot` accepts positions over the image's extent, as above.
- Fill values are numbers, NaN by default.

``benchmarks/bench_spizzle.py`` times spizzle against drizzle's C kernel
on a synthetic 2048x2048 exposure drizzled at 2x oversampling with
``pixfrac=0.8``.  On an Apple M3 Pro (11 threads), building ``W`` takes
0.26 s and `Coadd.add` about 0.012 s per plane: the science data with
three variance planes take 0.30 s, against 0.72 s for drizzle with the
variance planes in the same call, and each further pass over the same
geometry, such as a noise realization, costs about 0.015 s, against 0.56 s
for a drizzle call.  On a single thread spizzle is slower than drizzle,
and its peak memory is higher: 3.0 GiB for the exposure with three
variance planes, against 1.25 GiB.

Threads
-------

The geometry, `build_operator`, `Coadd.add` and `blot` run on numba's
threads, and `pixel_map` with ``stepsize=1`` on Python threads; set
numba's with ``numba.set_num_threads`` or the ``NUMBA_NUM_THREADS``
environment variable.  The results do not depend on the number of
threads.

Reference
---------

.. automodapi:: spizzle
   :no-inheritance-diagram:
