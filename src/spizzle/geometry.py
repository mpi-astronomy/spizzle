"""Overlap geometry of the drizzle square kernel.

Each input pixel, shrunk by ``pixfrac`` about its centre, is mapped to a
quadrilateral on the output grid by bilinear interpolation of the pixel
map, which holds the output coordinates of every input pixel centre.  The
area of that quadrilateral inside each output pixel it touches is the
geometry of the drizzle operator: it depends on the pixel map and
``pixfrac`` only, not on the data or the weights.

`corners` and `jacobian` are numpy ports of ``interpolate_four_points``,
``interpolate_point`` and the Jacobian in ``do_kernel_square`` of
spacetelescope/drizzle, with the same corner order and the same order of
floating point operations.  `overlap_triplets` runs drizzle's ``boxer``
and ``sgarea``, ported to numba, over every input pixel, in parallel over
input rows.  Unlike drizzle, which visits only the input pixels whose
centres fall inside the output frame, every input pixel with finite
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
        Output ``x`` and ``y`` coordinates of each input pixel centre.
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
    h = 0.5 * pixfrac
    out = np.empty((ny, nx, 4, 2))

    edge = np.ones((ny, nx), dtype=bool)
    edge[1:ny - 2, 1:nx - 2] = False
    j, i = np.nonzero(edge)
    x = i.astype(np.float64)
    y = j.astype(np.float64)
    for k, (sx, sy) in enumerate([(-1, 1), (1, 1), (1, -1), (-1, -1)]):
        out[j, i, k] = _interpolate_point(pixmap, x + sx * h, y + sy * h)

    if ny > 3 and nx > 3:
        out[1:ny - 2, 1:nx - 2] = _interpolate_four_points(pixmap, h)
    return out


def _interpolate_point(pixmap, xin, yin):
    """Bilinear interpolation of the pixel map at input points, as drizzle's ``interpolate_point``.

    Points beyond the outer pixel centres are extrapolated from the outer
    2x2 block.

    Parameters
    ----------
    pixmap : ndarray, shape (ny, nx, 2)
    xin, yin : ndarray, shape (n,)
        Input frame coordinates.

    Returns
    -------
    ndarray, shape (n, 2)
        Output ``x`` and ``y``.
    """
    ny, nx = pixmap.shape[:2]
    i0 = np.clip(np.trunc(xin), 0, nx - 2).astype(np.intp)
    j0 = np.clip(np.trunc(yin), 0, ny - 2).astype(np.intp)
    x = (xin - i0)[:, None]
    y = (yin - j0)[:, None]
    x1 = 1.0 - x
    y1 = 1.0 - y
    f00 = pixmap[j0, i0]
    f10 = pixmap[j0, i0 + 1]
    f01 = pixmap[j0 + 1, i0]
    f11 = pixmap[j0 + 1, i0 + 1]
    return f00 * x1 * y1 + f10 * x * y1 + f01 * x1 * y + f11 * x * y


def _interpolate_four_points(pixmap, h):
    """Corners of the interior input pixels, as drizzle's ``interpolate_four_points``.

    Each corner is the sum of four of the nine pixel map values around the
    pixel, each premultiplied by its bilinear weight.

    Parameters
    ----------
    pixmap : ndarray, shape (ny, nx, 2)
    h : float
        Half of ``pixfrac``.

    Returns
    -------
    ndarray, shape (ny - 3, nx - 3, 4, 2)
        Corners of input pixels ``1 <= y <= ny - 3``, ``1 <= x <= nx - 3``.
    """
    ny, nx = pixmap.shape[:2]

    def f(a, b):
        # pixel map at x offset a - 1 and y offset b - 1 from the pixel, times its weight
        if a == 1 and b == 1:
            fac = (1 - h) * (1 - h)
        elif a == 1 or b == 1:
            fac = (1 - h) * h
        else:
            fac = h * h
        return pixmap[b:ny - 3 + b, a:nx - 3 + a] * fac

    return np.stack([
        f(0, 1) + f(1, 1) + f(0, 2) + f(1, 2),
        f(1, 1) + f(2, 1) + f(1, 2) + f(2, 2),
        f(1, 0) + f(2, 0) + f(1, 1) + f(2, 1),
        f(0, 0) + f(1, 0) + f(0, 1) + f(1, 1),
    ], axis=2)


def jacobian(corners):
    """Area of each input pixel's quadrilateral on the output grid.

    Half the cross product of the diagonals, as in drizzle.  It is
    positive when the corners go clockwise on the output grid and negative
    when they go anticlockwise (a flip), and so are the overlap areas, so
    the sign cancels in the operator.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
        From `corners`.

    Returns
    -------
    ndarray, shape (ny, nx)
    """
    x = corners[..., 0]
    y = corners[..., 1]
    return 0.5 * ((x[..., 1] - x[..., 3]) * (y[..., 0] - y[..., 2]) - (x[..., 0] - x[..., 2]) * (y[..., 1] - y[..., 3]))


def overlap_triplets(corners, out_shape):
    """Overlap areas of input pixel quadrilaterals with output pixels.

    Every input pixel with finite corners is visited; output pixels outside
    ``out_shape`` are left out.  The triplets are in input pixel order (row
    by row) and, within an input pixel, in output pixel order, which is the
    order in which drizzle adds the contributions to each output pixel.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
        From `corners`.
    out_shape : tuple of int
        ``(ny_out, nx_out)``.

    Returns
    -------
    in_idx : ndarray of int64, shape (n,)
        Flat index of the input pixel, ``y * nx + x``.
    out_idx : ndarray of int64, shape (n,)
        Flat index of the output pixel, ``y * nx_out + x``.
    area : ndarray of float64, shape (n,)
        Overlap area in output pixels, non-zero; negative when the
        quadrilateral is anticlockwise.
    """
    corners = np.ascontiguousarray(corners, dtype=np.float64)
    if corners.ndim != 4 or corners.shape[2:] != (4, 2):
        raise ValueError(f"corners must have shape (ny, nx, 4, 2), not {corners.shape}")
    ny_out, nx_out = out_shape

    # each row writes its candidates into its own slice, sized by the bounding boxes
    sizes = _count_candidates(corners, ny_out, nx_out)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    total = int(sizes.sum())
    in_idx = np.empty(total, dtype=np.int64)
    out_idx = np.empty(total, dtype=np.int64)
    area = np.empty(total, dtype=np.float64)
    counts = _fill_triplets(corners, ny_out, nx_out, starts, in_idx, out_idx, area)

    new_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n = int(counts.sum())
    triplets = (np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64), np.empty(n, dtype=np.float64))
    _compact(starts, counts, new_starts, in_idx, out_idx, area, *triplets)
    return triplets


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
        Output pixel centre; the pixel spans ``is_ - 0.5`` to ``is_ + 0.5``.
    x0, y0, x1, y1, x2, y2, x3, y3 : float
        Quadrilateral corners in order.

    Returns
    -------
    float
        Positive for clockwise corners, negative for anticlockwise.
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
def _count_candidates(corners, ny_out, nx_out):
    """Number of output pixels in the bounding boxes of each input row.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
    ny_out, nx_out : int

    Returns
    -------
    ndarray of int64, shape (ny,)
    """
    ny, nx = corners.shape[0], corners.shape[1]
    sizes = np.zeros(ny, dtype=np.int64)
    for j in numba.prange(ny):
        n = 0
        for i in range(nx):
            ii0, ii1, jj0, jj1 = _output_box(corners, j, i, ny_out, nx_out)
            if ii1 >= ii0 and jj1 >= jj0:
                n += (ii1 - ii0 + 1) * (jj1 - jj0 + 1)
        sizes[j] = n
    return sizes


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _fill_triplets(corners, ny_out, nx_out, starts, in_idx, out_idx, area):
    """Write the non-zero overlaps of each input row from ``starts[row]`` on.

    Parameters
    ----------
    corners : ndarray, shape (ny, nx, 4, 2)
    ny_out, nx_out : int
    starts : ndarray of int64, shape (ny,)
        First slot of each row in the output arrays.
    in_idx, out_idx, area : ndarray
        Filled in place.

    Returns
    -------
    ndarray of int64, shape (ny,)
        Number of overlaps written for each row.
    """
    ny, nx = corners.shape[0], corners.shape[1]
    counts = np.zeros(ny, dtype=np.int64)
    for j in numba.prange(ny):
        n = starts[j]
        for i in range(nx):
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
        counts[j] = n - starts[j]
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
