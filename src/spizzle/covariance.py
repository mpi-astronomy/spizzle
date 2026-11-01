"""Covariance between nearby pixels of a drizzled image.

Drizzling spreads the noise of each input pixel over the output pixels it
lands on, so neighboring output pixels are correlated.  With independent
input noise of variance ``var_k`` in exposure k, the weighted sums of
`Coadd` have the covariance ``sum_k W_k diag(var_k) W_k.T``, and the image,
those sums over the weights,

    Cov(img[o], img[o']) = sum_k (W_k diag(var_k) W_k.T)[o, o'] / (wsum[o] wsum[o'])

An input pixel reaches only the few output pixels under its footprint, so
the covariance is zero beyond a small offset between ``o`` and ``o'``.
`Covariance` keeps it for the offsets up to ``radius`` in each coordinate,
one image per offset.  Entry ``(o, o')`` is the weighted dot product of
rows ``o`` and ``o'`` of ``W``, computed for every output pixel and offset
in parallel.
"""
import numba
import numpy as np

from .operator import _box_rows

__all__ = [
    "Covariance",
]


class Covariance:
    """Covariance between nearby pixels of the image of drizzled exposures, in float64.

    Add the same operators as to `Coadd`, each with the variance of its
    exposure's data (for example the sum of its variance planes); the image
    is `Coadd.out_img`.  The covariance is kept for the offsets ``(dy, dx)``
    with ``|dy|, |dx| <= radius``, for half of them: ``(0, 0)``, then
    ``(0, 1)`` to ``(0, radius)``, then ``(dy, -radius)`` to ``(dy, radius)``
    for ``dy`` from 1 to ``radius``.  The others follow from
    ``Cov(img[y, x], img[y - dy, x - dx]) = out_cov[k][y - dy, x - dx]``.
    Pixels farther apart than the footprint of an input pixel on the
    output grid share no input pixel and have no covariance, so it is
    complete for ``radius >= ceil(pixfrac * p * sqrt(2))``, with ``p`` the
    input pixel size in output pixels: 2 for ``p = 1`` and ``pixfrac = 1``,
    3 for ``p = 2`` and ``pixfrac = 0.8``.  It takes ``2 * radius *
    (radius + 1) + 1`` output-sized float64 images.

    Parameters
    ----------
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    radius : int
        Largest offset kept, in output pixels, in y and in x.
    fillval : float
        Value where the pixel or its neighbor has no weight, or the
        neighbor is outside the output grid.
    """

    def __init__(self, out_shape, radius=2, fillval=np.nan):
        if radius < 0:
            raise ValueError(f"radius must not be negative, not {radius}")
        self.out_shape = tuple(out_shape)
        self.radius = int(radius)
        self.fillval = fillval
        r = self.radius
        offsets = [(0, dx) for dx in range(r + 1)] + [(dy, dx) for dy in range(1, r + 1) for dx in range(-r, r + 1)]
        self.offsets = np.array(offsets, dtype=np.int64).reshape(-1, 2)
        n_out = self.out_shape[0] * self.out_shape[1]
        self._cnum = np.zeros((len(self.offsets), n_out))
        self._wsum = np.zeros(n_out)

    def add(self, W, var, iscale=1.0):
        """Add one exposure.

        Parameters
        ----------
        W : Operator or scipy.sparse array
            The exposure's operator, as for `Coadd.add`.
        var : ndarray, shape (ny, nx)
            Variance of the exposure's data.
        iscale : float
            Factor applied to the data; its square is applied to ``var``.
        """
        var = np.asarray(var, dtype=np.float64)
        matrix, y0, x0, box_ny, box_nx = _box_rows(W, self.out_shape, var.size)
        if not matrix.has_sorted_indices:
            matrix = matrix.sorted_indices()
        _add_covariance(matrix.indptr, matrix.indices, matrix.data, y0, x0, box_ny, box_nx, self.out_shape[1],
                        var.ravel(), iscale * iscale, self.offsets, self._cnum, self._wsum)

    @property
    def out_cov(self):
        """ndarray, shape (n_offsets,) + out_shape: plane k holds ``Cov(img[y, x], img[y + dy, x + dx])`` for ``offsets[k]``."""
        return np.array([self._plane(k) for k in range(len(self.offsets))])

    @property
    def out_var(self):
        """ndarray, shape out_shape: variance of the image, the plane of offset (0, 0)."""
        return self._plane(0)

    def _plane(self, k):
        """Covariance for ``offsets[k]``, normalized by the weights of both pixels."""
        ny, nx = self.out_shape
        dy, dx = self.offsets[k]
        wsum = self._wsum.reshape(ny, nx)
        out = np.full((ny, nx), self.fillval, dtype=np.float64)
        # pixels (y, x) whose neighbor (y + dy, x + dx) is on the grid
        here = (slice(0, ny - dy), slice(max(0, -dx), nx - max(0, dx)))
        there = (slice(dy, ny), slice(max(0, dx), nx - max(0, -dx)))
        w_here = wsum[here]
        w_there = wsum[there]
        good = (w_here != 0.0) & (w_there != 0.0)
        out[here][good] = self._cnum[k].reshape(ny, nx)[here][good] / (w_here[good] * w_there[good])
        return out


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _add_covariance(indptr, indices, values, y0, x0, box_ny, box_nx, nx_out, var, iscale2, offsets, cnum, wsum):
    """Add ``(W diag(var * iscale2) W.T)[o, o + offset]`` to ``cnum[k, o]`` for each offset k, and ``W @ 1`` to wsum.

    Each entry is the dot product of two rows of W weighted by the
    variance, found by merging the rows' sorted input indices.  Rows
    outside the box have no entries, so neighbors there add nothing.

    Parameters
    ----------
    indptr, indices, values : ndarray
        CSR arrays of W, one row per pixel of the box, indices sorted
        within each row.
    y0, x0, box_ny, box_nx : int
        First output row and column and shape of the box.
    nx_out : int
    var : ndarray, shape (n_in,)
    iscale2 : float
    offsets : ndarray of int64, shape (n_offsets, 2)
    cnum : ndarray of float64, shape (n_offsets, n_out)
    wsum : ndarray of float64, shape (n_out,)
    """
    for by in numba.prange(box_ny):
        for bx in range(box_nx):
            row = by * box_nx + bx
            lo = indptr[row]
            hi = indptr[row + 1]
            if lo == hi:
                continue
            o = (y0 + by) * nx_out + x0 + bx
            sw = 0.0
            for k in range(lo, hi):
                sw += values[k]
            wsum[o] += sw
            for m in range(offsets.shape[0]):
                ny = by + offsets[m, 0]
                nx = bx + offsets[m, 1]
                if ny < 0 or ny >= box_ny or nx < 0 or nx >= box_nx:
                    continue
                other = ny * box_nx + nx
                a = lo
                b = indptr[other]
                b_hi = indptr[other + 1]
                s = 0.0
                while a < hi and b < b_hi:
                    ia = indices[a]
                    ib = indices[b]
                    if ia == ib:
                        s += values[a] * values[b] * (var[ia] * iscale2)
                        a += 1
                        b += 1
                    elif ia < ib:
                        a += 1
                    else:
                        b += 1
                cnum[m, o] += s
