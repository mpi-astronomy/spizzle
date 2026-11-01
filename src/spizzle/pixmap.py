"""Pixel maps: the output pixel coordinates of every input pixel center.

`pixel_map` maps the input pixel grid of one WCS onto the pixel grid of
another through the shared astropy WCS interface (APE 14), as drizzle's
``calc_pixmap`` does.  Evaluating the WCS for every pixel is often the
slowest step of drizzling, so by default it evaluates them on a coarse
grid of input pixels and interpolates with bicubic splines: for the smooth
distortions of imaging detectors this reproduces the full evaluation to
about 1e-8 pixel (NIRCam, every 16th pixel) in a tenth of the time.  The
full evaluation, used for ``stepsize=1`` or when the coarse grid has
non-finite values, runs in blocks of rows on several threads.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
from scipy.interpolate import BSpline, RectBivariateSpline

__all__ = [
    "pixel_map",
]

_ROWS_PER_BLOCK = 64


def pixel_map(wcs_from, wcs_to, shape=None, stepsize=16, order=3, disable_bbox="to", threads=None):
    """Output pixel coordinates of every input pixel center, the pixel map of `drizzle_operator`.

    For each input pixel ``(y, x)`` of ``wcs_from``, the ``x`` and ``y``
    of the same world coordinates in ``wcs_to``, from
    ``wcs_to.world_to_pixel_values(*wcs_from.pixel_to_world_values(x, y))``
    as in drizzle's ``calc_pixmap``.  Both WCS must use the same celestial
    frame and units.

    With ``stepsize > 1`` the WCS are evaluated on a grid of input pixels
    at most ``stepsize`` apart, including the first and last, and the
    pixel map is interpolated between them with splines of degree
    ``order``.  This assumes that the transform is smooth; if any grid
    point has a non-finite result, every pixel is evaluated instead.

    Parameters
    ----------
    wcs_from, wcs_to : WCS
        Input and output WCS with the shared astropy interface, such as
        gwcs or astropy.wcs.
    shape : tuple of int, optional
        ``(ny, nx)`` of the input; by default ``wcs_from.array_shape``, or
        else the extent of its bounding box.
    stepsize : int
        Spacing of the evaluated grid in input pixels; 1 evaluates every
        pixel.
    order : {3, 1}
        Degree of the interpolating splines: 3 bicubic, 1 bilinear.
    disable_bbox : {"to", "from", "both", "none"}
        Which bounding boxes to switch off while evaluating, as in
        drizzle's ``calc_pixmap``.  Pixels outside an input bounding box
        that stays on are NaN.
    threads : int, optional
        Threads for evaluating every pixel; numba's number of threads by
        default.  The results do not depend on it.

    Returns
    -------
    ndarray, shape (ny, nx, 2)
        Output ``x`` and ``y`` of each input pixel.
    """
    if order not in (1, 3):
        raise ValueError(f"order must be 1 or 3, not {order}")
    if disable_bbox not in ("to", "from", "both", "none"):
        raise ValueError(f"disable_bbox must be 'to', 'from', 'both' or 'none', not {disable_bbox!r}")
    if threads is None:
        threads = numba.get_num_threads()

    bbox_from = _bounding_box(wcs_from)
    bbox_to = _bounding_box(wcs_to)
    if shape is None:
        shape = wcs_from.array_shape
    if shape is None and bbox_from is not None:
        shape = tuple(math.ceil(hi + 0.5) for lo, hi in bbox_from[::-1])
    if shape is None:
        raise ValueError("shape is needed when wcs_from has neither array_shape nor a bounding box")
    ny, nx = shape

    # the input pixels inside the bounding box that stays on
    x0, x1, y0, y1 = 0, nx - 1, 0, ny - 1
    if bbox_from is not None and disable_bbox in ("to", "none"):
        (xlo, xhi), (ylo, yhi) = bbox_from
        x0, x1 = max(x0, math.ceil(xlo)), min(x1, math.floor(xhi))
        y0, y1 = max(y0, math.ceil(ylo)), min(y1, math.floor(yhi))

    if disable_bbox in ("from", "both") and bbox_from is not None:
        wcs_from.bounding_box = None
    if disable_bbox in ("to", "both") and bbox_to is not None:
        wcs_to.bounding_box = None
    try:
        pixmap = None
        if stepsize > 1 and min(x1 - x0, y1 - y0) >= order:
            pixmap = _interpolated(wcs_from, wcs_to, shape, (x0, x1, y0, y1), stepsize, order)
        if pixmap is None:
            pixmap = _evaluated(wcs_from, wcs_to, shape, threads)
    finally:
        if disable_bbox in ("from", "both") and bbox_from is not None:
            wcs_from.bounding_box = bbox_from
        if disable_bbox in ("to", "both") and bbox_to is not None:
            wcs_to.bounding_box = bbox_to
    return pixmap


def _bounding_box(wcs):
    """The WCS's bounding box as ``((xmin, xmax), (ymin, ymax))``, or None.

    Accepts an astropy ``ModelBoundingBox``, as gwcs has, or a tuple of
    intervals in ``(x, y)`` order.
    """
    bbox = getattr(wcs, "bounding_box", None)
    if bbox is None:
        return None
    try:
        bbox = bbox.bounding_box(order="F")
    except AttributeError:
        pass
    return tuple(tuple(interval) for interval in bbox)


def _interpolated(wcs_from, wcs_to, shape, box, stepsize, order):
    """Pixel map from the WCS on a coarse grid of the pixels in ``box``, interpolated with splines.

    Parameters
    ----------
    wcs_from, wcs_to, shape, stepsize, order
        As for `pixel_map`.
    box : tuple of int
        ``(x0, x1, y0, y1)``, inclusive range of input pixels to fill;
        the others are NaN.

    Returns
    -------
    ndarray, shape (ny, nx, 2), or None if a grid point is not finite
    """
    x0, x1, y0, y1 = box
    # at most stepsize apart, and enough points for splines of degree order
    nx_grid = max(math.ceil((x1 - x0) / stepsize) + 1, order + 1)
    ny_grid = max(math.ceil((y1 - y0) / stepsize) + 1, order + 1)
    x_grid = np.linspace(x0, x1, nx_grid)
    y_grid = np.linspace(y0, y1, ny_grid)
    y, x = np.meshgrid(y_grid, x_grid, indexing="ij")
    xo, yo = wcs_to.world_to_pixel_values(*wcs_from.pixel_to_world_values(x, y))
    if not (np.all(np.isfinite(xo)) and np.all(np.isfinite(yo))):
        return None

    x_all = np.arange(x0, x1 + 1, dtype=np.float64)
    y_all = np.arange(y0, y1 + 1, dtype=np.float64)
    if (x0, x1, y0, y1) == (0, shape[1] - 1, 0, shape[0] - 1):
        pixmap = np.empty(tuple(shape) + (2,))
    else:
        pixmap = np.full(tuple(shape) + (2,), np.nan)
    for q, values in enumerate((xo, yo)):
        pixmap[y0:y1 + 1, x0:x1 + 1, q] = _spline_on_grid(y_grid, x_grid, values, order, y_all, x_all)
    return pixmap


def _spline_on_grid(y_grid, x_grid, values, order, y_all, x_all):
    """Interpolating spline through ``values`` on a grid, evaluated on another grid.

    The tensor-product spline on the grid ``y_all`` by ``x_all`` is
    ``By @ C @ Bx.T``, with C its coefficients and By, Bx the B-spline
    basis matrices of ``y_all`` and ``x_all``, which have ``order + 1``
    non-zero entries per row.  As sparse matrix products this is several
    times faster than scipy's evaluation of the same spline.

    Parameters
    ----------
    y_grid, x_grid : ndarray
        Increasing coordinates of the grid of ``values``.
    values : ndarray, shape (y_grid.size, x_grid.size)
    order : int
        Degree of the spline in each coordinate.
    y_all, x_all : ndarray
        Increasing coordinates to evaluate at, within the grid.

    Returns
    -------
    ndarray, shape (y_all.size, x_all.size)
    """
    ty, tx, c = RectBivariateSpline(y_grid, x_grid, values, kx=order, ky=order).tck
    coefficients = c.reshape(ty.size - order - 1, tx.size - order - 1)
    by = BSpline.design_matrix(y_all, ty, order)
    bx = BSpline.design_matrix(x_all, tx, order)
    # By @ (C @ Bx.T), so that the result comes out in row order
    return by @ np.ascontiguousarray((bx @ coefficients.T).T)


def _evaluated(wcs_from, wcs_to, shape, threads):
    """Pixel map from the WCS at every input pixel, in blocks of rows on ``threads`` threads.

    Each pixel's result is computed from that pixel alone, so it does not
    depend on the blocks or the threads.

    Returns
    -------
    ndarray, shape (ny, nx, 2)
    """
    ny, nx = shape
    pixmap = np.empty((ny, nx, 2))

    def block(j0):
        j1 = min(j0 + _ROWS_PER_BLOCK, ny)
        y, x = np.mgrid[j0:j1, 0:nx].astype(np.float64)
        xo, yo = wcs_to.world_to_pixel_values(*wcs_from.pixel_to_world_values(x, y))
        pixmap[j0:j1, :, 0] = xo
        pixmap[j0:j1, :, 1] = yo

    starts = range(0, ny, _ROWS_PER_BLOCK)
    if threads > 1:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(block, starts))
    else:
        for j0 in starts:
            block(j0)
    return pixmap
