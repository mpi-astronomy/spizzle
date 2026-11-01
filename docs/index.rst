spizzle
=======

Drizzle's square kernel written out as an explicit sparse linear operator.
Drizzling an exposure is the linear map ``W @ data``, and ``W`` depends on
the pixel map, ``pixfrac`` and the input weights, not on the data.  spizzle
builds ``W`` once per exposure and applies it to the science data, the
weights and every variance plane, both in parallel, summing in float64 and
dividing once at the end.  The README of the repository describes how the
results compare with drizzle.

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

``op`` is an `Operator`: ``W`` on the exposure's bounding box of the output
grid, ``op.box``, so that building and applying it scale with the exposure
rather than with the output grid.

The pixel map holds the output ``x`` and ``y`` of every input pixel
center, shape ``(ny, nx, 2)``, NaN where an input pixel has no mapping.
`pixel_map` makes it from two WCS with the shared astropy interface, by
default from the WCS on every 16th pixel, interpolated with bicubic
splines.

Reference
---------

.. automodapi:: spizzle
   :no-inheritance-diagram:
