"""Blot: a resampled image interpolated back onto the pixel grid of an exposure.

`blot` ports the bilinear interpolation of spacetelescope/drizzle's
``blot_image`` (``interpolate_bilinear`` and the loop in ``doblot``),
with the same treatment of the last row and column and the same order of
floating point operations, so that it matches drizzle bit for bit.  It is
the interpolation stcal's ``gwcs_blot`` uses in outlier detection.

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
        Value of the pixels whose position is NaN or outside ``data``.
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

    As ``doblot``: positions that are NaN or outside ``[0, nx_data) x [0,
    ny_data)`` get ``fillval``.  Each pixel depends on its own position
    only, so the rows run in parallel.
    """
    ny_data, nx_data = data.shape
    ny, nx = out.shape
    for j in numba.prange(ny):
        for i in range(nx):
            x = pixmap[j, i, 0]
            y = pixmap[j, i, 1]
            # False for NaN
            if x >= 0.0 and x < nx_data and y >= 0.0 and y < ny_data:
                out[j, i] = _bilinear(data, x, y) * iscale
            else:
                out[j, i] = fillval


@numba.njit(cache=True, error_model="numpy")
def _bilinear(data, x, y):
    """drizzle's ``interpolate_bilinear`` at ``(x, y)``, inside the image.

    The last column interpolates along y only, the last row along x only,
    and the last pixel is its own value.
    """
    ny_data, nx_data = data.shape
    i = int(x)
    j = int(y)
    f00 = np.float64(data[j, i])
    if i == nx_data - 1:
        if j == ny_data - 1:
            return f00
        sy = y - j
        return (1.0 - sy) * f00 + sy * data[j + 1, i]
    if j == ny_data - 1:
        sx = x - i
        return (1.0 - sx) * f00 + sx * data[j, i + 1]
    sx = x - i
    tx = 1.0 - sx
    sy = y - j
    ty = 1.0 - sy
    return tx * ty * f00 + sx * ty * data[j, i + 1] + sy * tx * data[j + 1, i] + sx * sy * data[j + 1, i + 1]
