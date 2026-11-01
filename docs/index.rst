spizzle
=======

Drizzle's square kernel written out as an explicit sparse linear operator.
Drizzling an exposure is the linear map ``W @ data``, and ``W`` depends on
the pixel map, ``pixfrac`` and the input weights, not on the data.  spizzle
builds ``W`` once per exposure, with the overlap geometry computed in
parallel, and applies it to the science data, the weights and every
variance plane, summing in float64 and dividing once at the end.  The
README of the repository describes how the results compare with drizzle.

Usage
-----

.. code-block:: python

    from spizzle import Coadd, drizzle_operator

    coadd = Coadd(out_shape, n_var=1)
    for data, var, weights, pixmap in exposures:
        W = drizzle_operator(pixmap, out_shape, pixfrac=0.8, weights=weights)
        coadd.add(W, data, variances=[var])

    image = coadd.out_img
    weight = coadd.out_wht
    variance = coadd.out_var[0]

The pixel map holds the output ``x`` and ``y`` of every input pixel
centre, shape ``(ny, nx, 2)``, NaN where an input pixel has no mapping.

Reference
---------

.. automodapi:: spizzle
   :no-inheritance-diagram:
