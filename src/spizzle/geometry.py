"""Overlap geometry of the drizzle square kernel.

Each input pixel, shrunk by ``pixfrac`` about its center, is mapped to a
quadrilateral on the output grid by bilinear interpolation of the pixel
map, which holds the output coordinates of every input pixel center.  The
area of that quadrilateral inside each output pixel it touches is the
geometry of the drizzle operator: it depends on the pixel map and
``pixfrac`` only, not on the data or the weights.

`corners` and `jacobian` port ``interpolate_four_points``,
``interpolate_point`` and the Jacobian in ``do_kernel_square`` of
spacetelescope/drizzle, with the same corner order and the same order of
floating point operations, and `overlap_triplets` runs drizzle's ``boxer``
and ``sgarea`` over every input pixel.  All three are compiled with numba
and run in parallel over input rows.  Unlike drizzle, which visits only the
input pixels whose
centers fall inside the output frame, every input pixel with finite
corners is visited, so output pixels at the edges get all of their
overlap.

The ported functions are derived from drizzle, Copyright (C) 2011,2014
Association of Universities for Research in Astronomy (AURA), under the
BSD 3-clause license in ``licenses/DRIZZLE_LICENSE.rst``.
"""
import numba
import numpy as np

__all__ = [
    "corners",
    "jacobian",
    "overlap_triplets",
]

_MAX_DOUBLE = 1.7976931348623158e308


def corners(pixmap, pixfrac=1.0):
    """Output coordinates of the corners of every shrunken input pixel.

    The corners are at ``(x - h, y + h)``, ``(x + h, y + h)``,
    ``(x + h, y - h)`` and ``(x - h, y - h)`` in the input frame, in that
    order, with ``h = pixfrac / 2``.  Interior pixels use drizzle's
    nine-point interpolation and pixels within one pixel of the left and
    bottom edges, or two pixels of the right and top edges, use bilinear
    interpolation of each corner, extrapolating at the image edge, as
    drizzle does.

    Parameters
    ----------
    pixmap : ndarray, shape (ny, nx, 2)
        Output ``x`` and ``y`` coordinates of each input pixel center.
        NaN marks pixels without a mapping.
    pixfrac : float
        Linear size of the shrunken pixel as a fraction of the input pixel.

    Returns
    -------
    ndarray, shape (ny, nx, 4, 2)
        Output ``x`` and ``y`` of the four corners of each input pixel; NaN
        where an interpolation used a NaN in the pixel map.
    """
    pixmap = np.asarray(pixmap, dtype=np.float64)
    if pixmap.ndim != 3 or pixmap.shape[2] != 2:
        raise ValueError(f"pixmap must have shape (ny, nx, 2), not {pixmap.shape}")
    ny, nx = pixmap.shape[:2]
    if ny < 2 or nx < 2:
        raise ValueError(f"pixmap must be at least 2x2, not {ny}x{nx}")
    if not pixfrac > 0.001:
        raise ValueError(f"pixfrac must be larger than 0.001 (drizzle switches to the point kernel), not {pixfrac}")
    return _corners(pixmap, 0.5 * pixfrac)


def jacobian(corners):
    """Area of each input pixel's quadrilateral on the output grid.

    Half the cross product of the diagonals, as in drizzle.  It is
    positive when the corners go clockwise on the output grid and negative
    when they go counterclockwise (a flip), and so are the overlap areas, so
    the sign cancels in the operator.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
        From `corners`.

    Returns
    -------
    ndarray, shape (ny, nx)
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.ndim != 4 or corners.shape[2:] != (4, 2):
        raise ValueError(f"corners must have shape (ny, nx, 4, 2), not {corners.shape}")
    return _jacobian(corners)


def overlap_triplets(corners, out_shape, region=None):
    """Overlap areas of input pixel quadrilaterals with output pixels.

    Every input pixel with finite corners is visited, or only those in
    ``region``; output pixels outside ``out_shape`` are left out.  The
    triplets are in input pixel order (row by row) and, within an input
    pixel, in output pixel order, which is the order in which drizzle adds
    the contributions to each output pixel.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
        From `corners`.
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    region : tuple of slice, optional
        ``(slice(y0, y1), slice(x0, x1))``, the input pixels to visit, as
        drizzle's ``xmin`` to ``xmax`` and ``ymin`` to ``ymax`` with
        ``x1 = xmax + 1`` and ``y1 = ymax + 1``; by default all of them.

    Returns
    -------
    in_idx : ndarray of int64, shape (n,)
        Flat index of the input pixel, ``y * nx + x``.
    out_idx : ndarray of int64, shape (n,)
        Flat index of the output pixel, ``y * nx_out + x``.
    area : ndarray of float64, shape (n,)
        Overlap area in output pixels, non-zero; negative when the
        quadrilateral is counterclockwise.
    """
    corners = np.ascontiguousarray(corners, dtype=np.float64)
    if corners.ndim != 4 or corners.shape[2:] != (4, 2):
        raise ValueError(f"corners must have shape (ny, nx, 4, 2), not {corners.shape}")
    ny_out, nx_out = out_shape
    j0, j1, i0, i1 = _region_bounds(region, corners.shape[:2])

    # each row writes its candidates into its own slice, sized by the bounding boxes
    sizes = _count_candidates(corners, j0, j1, i0, i1, ny_out, nx_out)
    starts = np.cumsum(sizes) - sizes
    total = int(sizes.sum())
    in_idx = np.empty(total, dtype=np.int64)
    out_idx = np.empty(total, dtype=np.int64)
    area = np.empty(total, dtype=np.float64)
    counts = _fill_triplets(corners, j0, j1, i0, i1, ny_out, nx_out, starts, in_idx, out_idx, area)

    new_starts = np.cumsum(counts) - counts
    n = int(counts.sum())
    triplets = (np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64), np.empty(n, dtype=np.float64))
    _compact(starts, counts, new_starts, in_idx, out_idx, area, *triplets)
    return triplets


def _region_bounds(region, shape):
    """``(j0, j1, i0, i1)``, the rows ``j0:j1`` and columns ``i0:i1`` of ``region`` within an input of ``shape``."""
    ny, nx = shape
    if region is None:
        return 0, ny, 0, nx
    if len(region) != 2:
        raise ValueError(f"region must be a pair of slices, not {region!r}")
    bounds = []
    for s, n in zip(region, shape):
        if not isinstance(s, slice):
            raise ValueError(f"region must be a pair of slices, not {region!r}")
        start, stop, step = s.indices(n)
        if step != 1:
            raise ValueError(f"region slices must have step 1, not {step}")
        bounds += [start, max(start, stop)]
    return tuple(bounds)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _corners(pixmap, h):
    """Corners of every input pixel shrunk to half-width ``h``; see `corners`.

    Parameters
    ----------
    pixmap : ndarray, shape (ny, nx, 2)
    h : float
        Half of ``pixfrac``.

    Returns
    -------
    ndarray, shape (ny, nx, 4, 2)
    """
    ny, nx = pixmap.shape[0], pixmap.shape[1]
    out = np.empty((ny, nx, 4, 2))
    for j in numba.prange(ny):
        for i in range(nx):
            if 1 <= j <= ny - 3 and 1 <= i <= nx - 3:
                _interpolate_four_points(pixmap, j, i, h, out)
            else:
                x = float(i)
                y = float(j)
                _interpolate_point(pixmap, x - h, y + h, out, j, i, 0)
                _interpolate_point(pixmap, x + h, y + h, out, j, i, 1)
                _interpolate_point(pixmap, x + h, y - h, out, j, i, 2)
                _interpolate_point(pixmap, x - h, y - h, out, j, i, 3)
    return out


@numba.njit(cache=True, error_model="numpy")
def _interpolate_point(pixmap, xin, yin, out, j, i, k):
    """Bilinear interpolation of the pixel map at one input point, asdrizzle's ``interpolate_point``.

    Points beyond the outer pixel centers are extrapolated from the outer
    2x2 block.

    Parameters
    ----------
    pixmap : ndarray, shape (ny, nx, 2)
    xin, yin : float
        Input frame coordinates.
    out : ndarray, shape (ny, nx, 4, 2)
        Output ``x`` and ``y`` are written to ``out[j, i, k]``.
    j, i, k : int
        Input pixel and corner.
    """
    ny, nx = pixmap.shape[0], pixmap.shape[1]
    i0 = int(min(max(np.trunc(xin), 0.0), nx - 2.0))
    j0 = int(min(max(np.trunc(yin), 0.0), ny - 2.0))
    x = xin - i0
    y = yin - j0
    x1 = 1.0 - x
    y1 = 1.0 - y
    for q in range(2):
        out[j, i, k, q] = (
            pixmap[j0, i0, q] * x1 * y1 + pixmap[j0, i0 + 1, q] * x * y1
            + pixmap[j0 + 1, i0, q] * x1 * y + pixmap[j0 + 1, i0 + 1, q] * x * y
        )


@numba.njit(cache=True, error_model="numpy")
def _interpolate_four_points(pixmap, j, i, h, out):
    """Corners of interior input pixel ``(j, i)``, as drizzle's ``interpolate_four_points``.

    Each corner is the sum of four of the nine pixel map values around the
    pixel, each premultiplied by its bilinear weight.

    Parameters
    ----------
    pixmap : ndarray, shape (ny, nx, 2)
    j, i : int
        Input pixel, with ``1 <= j <= ny - 3`` and ``1 <= i <= nx - 3``.
    h : float
        Half of ``pixfrac``.
    out : ndarray, shape (ny, nx, 4, 2)
        The corners are written to ``out[j, i]``.
    """
    center = (1.0 - h) * (1.0 - h)
    side = (1.0 - h) * h
    diagonal = h * h
    for q in range(2):
        # p<a><b>: pixel map at x offset a - 1 and y offset b - 1 from the pixel
        p00 = pixmap[j - 1, i - 1, q]
        p10 = pixmap[j - 1, i, q]
        p20 = pixmap[j - 1, i + 1, q]
        p01 = pixmap[j, i - 1, q]
        p11 = pixmap[j, i, q]
        p21 = pixmap[j, i + 1, q]
        p02 = pixmap[j + 1, i - 1, q]
        p12 = pixmap[j + 1, i, q]
        p22 = pixmap[j + 1, i + 1, q]
        out[j, i, 0, q] = p01 * side + p11 * center + p02 * diagonal + p12 * side
        out[j, i, 1, q] = p11 * center + p21 * side + p12 * side + p22 * diagonal
        out[j, i, 2, q] = p10 * side + p20 * diagonal + p11 * center + p21 * side
        out[j, i, 3, q] = p00 * diagonal + p10 * side + p01 * side + p11 * center


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _jacobian(corners):
    """Half the cross product of the diagonals of each quadrilateral; see `jacobian`.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)

    Returns
    -------
    ndarray, shape (ny, nx)
    """
    ny, nx = corners.shape[0], corners.shape[1]
    out = np.empty((ny, nx))
    for j in numba.prange(ny):
        for i in range(nx):
            x0, y0 = corners[j, i, 0, 0], corners[j, i, 0, 1]
            x1, y1 = corners[j, i, 1, 0], corners[j, i, 1, 1]
            x2, y2 = corners[j, i, 2, 0], corners[j, i, 2, 1]
            x3, y3 = corners[j, i, 3, 0], corners[j, i, 3, 1]
            out[j, i] = 0.5 * ((x1 - x3) * (y0 - y2) - (x0 - x2) * (y1 - y3))
    return out


@numba.njit(cache=True, error_model="numpy")
def _nintd(x):
    """Round to the nearest integer, halves away from zero, as C ``round``.

    numpy and Python round halves to even.

    Parameters
    ----------
    x : float

    Returns
    -------
    float
    """
    t = np.trunc(x)
    f = x - t
    if f >= 0.5:
        t += 1.0
    elif f <= -0.5:
        t -= 1.0
    return t


@numba.njit(cache=True, error_model="numpy")
def _output_box(corners, j, i, ny_out, nx_out):
    """Output pixel range that the quadrilateral of input pixel ``(j, i)`` can overlap.

    Rounded and clipped to the output as in drizzle's ``do_kernel_square``.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
    j, i : int
        Input pixel.
    ny_out, nx_out : int
        Output shape.

    Returns
    -------
    ii0, ii1, jj0, jj1 : int
        Inclusive x and y ranges of output pixels; empty (``ii0 > ii1``)
        when a corner is not finite.
    """
    xmin = _MAX_DOUBLE
    xmax = -_MAX_DOUBLE
    ymin = _MAX_DOUBLE
    ymax = -_MAX_DOUBLE
    for k in range(4):
        x = corners[j, i, k, 0]
        y = corners[j, i, k, 1]
        if not (np.isfinite(x) and np.isfinite(y)):
            return 0, -1, 0, -1
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    # clip before converting, so that far-off pixels cannot overflow an int
    ii0 = int(max(_nintd(xmin), 0.0))
    ii1 = int(min(_nintd(xmax), nx_out - 1.0))
    jj0 = int(max(_nintd(ymin), 0.0))
    jj1 = int(min(_nintd(ymax), ny_out - 1.0))
    return ii0, ii1, jj0, jj1


@numba.njit(cache=True, error_model="numpy")
def _sgarea(x1, y1, x2, y2):
    """Signed area under a line segment within the unit square at the origin, as drizzle's ``sgarea``.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Segment end points.

    Returns
    -------
    float
        Positive for a segment going in +x, negative in -x.
    """
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0.0:
        return 0.0

    if dx < 0.0:
        sgn_dx = -1.0
        xlo = x2
        xhi = x1
    else:
        sgn_dx = 1.0
        xlo = x1
        xhi = x2

    if xlo >= 1.0 or xhi <= 0.0:
        return 0.0

    xlo = xlo if xlo > 0.0 else 0.0
    xhi = xhi if xhi < 1.0 else 1.0

    slope = dy / dx
    ylo = y1 + slope * (xlo - x1)
    yhi = y1 + slope * (xhi - x1)

    if ylo <= 0.0 and yhi <= 0.0:
        return 0.0

    if ylo >= 1.0 and yhi >= 1.0:
        return sgn_dx * (xhi - xlo)

    det = x1 * y2 - y1 * x2

    if ylo < 0.0:
        ylo = 0.0
        xlo = det / dy

    if yhi < 0.0:
        yhi = 0.0
        xhi = det / dy

    if ylo <= 1.0:
        if yhi <= 1.0:
            return sgn_dx * 0.5 * (xhi - xlo) * (yhi + ylo)
        xtop = (dx + det) / dy
        return sgn_dx * (0.5 * (xtop - xlo) * (1.0 + ylo) + xhi - xtop)

    xtop = (dx + det) / dy
    return sgn_dx * (0.5 * (xhi - xtop) * (1.0 + yhi) + xtop - xlo)


@numba.njit(cache=True, error_model="numpy")
def _boxer(is_, js, x0, y0, x1, y1, x2, y2, x3, y3):
    """Area of a quadrilateral inside output pixel ``(js, is_)``, as drizzle's ``boxer``.

    Parameters
    ----------
    is_, js : float
        Output pixel center; the pixel spans ``is_ - 0.5`` to ``is_ + 0.5``.
    x0, y0, x1, y1, x2, y2, x3, y3 : float
        Quadrilateral corners in order.

    Returns
    -------
    float
        Positive for clockwise corners, negative for counterclockwise.
    """
    is_ -= 0.5
    js -= 0.5
    px0 = x0 - is_
    py0 = y0 - js
    px1 = x1 - is_
    py1 = y1 - js
    px2 = x2 - is_
    py2 = y2 - js
    px3 = x3 - is_
    py3 = y3 - js
    total = 0.0
    total += _sgarea(px0, py0, px1, py1)
    total += _sgarea(px1, py1, px2, py2)
    total += _sgarea(px2, py2, px3, py3)
    total += _sgarea(px3, py3, px0, py0)
    return total


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _count_candidates(corners, j0, j1, i0, i1, ny_out, nx_out):
    """Number of output pixels in the bounding boxes of each input row of the region.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
    j0, j1, i0, i1 : int
        Rows ``j0:j1`` and columns ``i0:i1`` of the input to visit.
    ny_out, nx_out : int

    Returns
    -------
    ndarray of int64, shape (j1 - j0,)
    """
    sizes = np.zeros(j1 - j0, dtype=np.int64)
    for r in numba.prange(j1 - j0):
        j = j0 + r
        n = 0
        for i in range(i0, i1):
            ii0, ii1, jj0, jj1 = _output_box(corners, j, i, ny_out, nx_out)
            if ii1 >= ii0 and jj1 >= jj0:
                n += (ii1 - ii0 + 1) * (jj1 - jj0 + 1)
        sizes[r] = n
    return sizes


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _fill_triplets(corners, j0, j1, i0, i1, ny_out, nx_out, starts, in_idx, out_idx, area):
    """Write the non-zero overlaps of each input row of the region from ``starts[row - j0]`` on.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
    j0, j1, i0, i1 : int
        Rows ``j0:j1`` and columns ``i0:i1`` of the input to visit.
    ny_out, nx_out : int
    starts : ndarray of int64, shape (j1 - j0,)
        First slot of each row in the output arrays.
    in_idx, out_idx, area : ndarray
        Filled in place.

    Returns
    -------
    ndarray of int64, shape (j1 - j0,)
        Number of overlaps written for each row.
    """
    nx = corners.shape[1]
    counts = np.zeros(j1 - j0, dtype=np.int64)
    for r in numba.prange(j1 - j0):
        j = j0 + r
        n = starts[r]
        for i in range(i0, i1):
            ii0, ii1, jj0, jj1 = _output_box(corners, j, i, ny_out, nx_out)
            x0 = corners[j, i, 0, 0]
            y0 = corners[j, i, 0, 1]
            x1 = corners[j, i, 1, 0]
            y1 = corners[j, i, 1, 1]
            x2 = corners[j, i, 2, 0]
            y2 = corners[j, i, 2, 1]
            x3 = corners[j, i, 3, 0]
            y3 = corners[j, i, 3, 1]
            for jj in range(jj0, jj1 + 1):
                for ii in range(ii0, ii1 + 1):
                    a = _boxer(float(ii), float(jj), x0, y0, x1, y1, x2, y2, x3, y3)
                    if a != 0.0:
                        in_idx[n] = j * nx + i
                        out_idx[n] = jj * nx_out + ii
                        area[n] = a
                        n += 1
        counts[r] = n - starts[r]
    return counts


@numba.njit(parallel=True, cache=True)
def _compact(starts, counts, new_starts, in_idx, out_idx, area, in_out, out_out, area_out):
    """Copy each row's overlaps from its candidate slice to consecutive output slots.

    Parameters
    ----------
    starts, counts, new_starts : ndarray of int64, shape (ny,)
        Slice start and length of each row in the candidate arrays, and its
        start in the output arrays.
    in_idx, out_idx, area : ndarray
        Candidate arrays.
    in_out, out_out, area_out : ndarray
        Output arrays, filled in place.
    """
    for j in numba.prange(starts.size):
        for t in range(counts[j]):
            in_out[new_starts[j] + t] = in_idx[starts[j] + t]
            out_out[new_starts[j] + t] = out_idx[starts[j] + t]
            area_out[new_starts[j] + t] = area[starts[j] + t]
