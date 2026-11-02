"""Blot: a resampled image interpolated back onto the pixel grid of an exposure.

`blot` ports the bilinear interpolation of spacetelescope/drizzle's
``blot_image`` (``interpolate_bilinear`` and the loop in ``doblot``),
with the same order of floating point operations, so that it matches
drizzle bit for bit inside the outermost pixel centers.  It is the
interpolation stcal's ``gwcs_blot`` uses in outlier detection.

Pixel centers are at integer coordinates, so an image of ``nx`` columns
covers ``[-0.5, nx - 0.5)``.  drizzle accepts positions in ``[0, nx)``
instead: it fills the outer half of the first row and column, and gives
values up to half a pixel beyond the last edge.  `blot` accepts the
image's extent, and between the outermost pixel centers and the edge it
holds the edge values, as drizzle does in its last row and column.

The ported functions are derived from drizzle, Copyright (C) 2011,2014
Association of Universities for Research in Astronomy (AURA), under the
BSD 3-clause license in ``licenses/DRIZZLE_LICENSE.rst``.
"""
import numba
import numpy as np

__all__ = [
    "blot",
]


def blot(data, pixmap, fillval=np.nan, iscale=1.0, out=None):
    """Bilinear interpolation of ``data`` at the positions in ``pixmap``, as drizzle's ``blot_image``.

    Positions within half a pixel of an edge, beyond the outermost pixel
    centers, get the edge values, interpolated along the edge.  Unlike
    drizzle's, the valid positions are the image's extent, ``[-0.5,
    nx_data - 0.5) x [-0.5, ny_data - 0.5)``.

    Parameters
    ----------
    data : ndarray, shape (ny_data, nx_data)
        Image to interpolate, such as a mosaic or the median of resampled
        exposures; float32 or float64.
    pixmap : ndarray, shape (ny, nx, 2)
        For each pixel of the blotted image, the ``x`` and ``y`` of its
        center in ``data``: the pixel map of the exposure onto the grid of
        ``data``, as for `drizzle_operator`.
    fillval : float
        Value of the pixels whose position is NaN or outside the extent
        of ``data``.
    iscale : float
        Factor applied to the interpolated values.
    out : ndarray, shape (ny, nx), optional
        float32 or float64 array to write the result to.

    Returns
    -------
    ndarray, shape (ny, nx)
        ``out``, or a new float64 array.
    """
    data = np.asarray(data)
    pixmap = np.asarray(pixmap, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, not of shape {data.shape}")
    if pixmap.ndim != 3 or pixmap.shape[2] != 2:
        raise ValueError(f"pixmap must have shape (ny, nx, 2), not {pixmap.shape}")
    if out is None:
        out = np.empty(pixmap.shape[:2])
    elif out.shape != pixmap.shape[:2]:
        raise ValueError(f"out has shape {out.shape}, the pixel map {pixmap.shape[:2]}")
    _blot(data, pixmap, float(fillval), float(iscale), out)
    return out


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _blot(data, pixmap, fillval, iscale, out):
    """Write the bilinear interpolation of ``data`` at each pixel's position, times ``iscale``, to ``out``.

    As ``doblot``, but positions that are NaN or outside ``[-0.5, nx_data
    - 0.5) x [-0.5, ny_data - 0.5)`` get ``fillval``.  Each pixel depends
    on its own position only, so the rows run in parallel.
    """
    ny_data, nx_data = data.shape
    ny, nx = out.shape
    for j in numba.prange(ny):
        for i in range(nx):
            x = pixmap[j, i, 0]
            y = pixmap[j, i, 1]
            # False for NaN
            if x >= -0.5 and x < nx_data - 0.5 and y >= -0.5 and y < ny_data - 0.5:
                out[j, i] = _bilinear(data, x, y) * iscale
            else:
                out[j, i] = fillval


@numba.njit(cache=True, error_model="numpy")
def _bilinear(data, x, y):
    """drizzle's ``interpolate_bilinear`` at ``(x, y)``, inside the image's extent.

    Beyond the outermost pixel centers, the first and last columns
    interpolate along y only, the first and last rows along x only, and
    the corner pixels are their own value.
    """
    ny_data, nx_data = data.shape
    # truncation toward zero: 0 for positions in [-0.5, 0)
    i = int(x)
    j = int(y)
    edge_x = x < 0.0 or i == nx_data - 1
    edge_y = y < 0.0 or j == ny_data - 1
    f00 = np.float64(data[j, i])
    if edge_x:
        if edge_y:
            return f00
        sy = y - j
        return (1.0 - sy) * f00 + sy * data[j + 1, i]
    if edge_y:
        sx = x - i
        return (1.0 - sx) * f00 + sx * data[j, i + 1]
    sx = x - i
    tx = 1.0 - sx
    sy = y - j
    ty = 1.0 - sy
    return tx * ty * f00 + sx * ty * data[j, i + 1] + sy * tx * data[j + 1, i] + sx * sy * data[j + 1, i + 1]
