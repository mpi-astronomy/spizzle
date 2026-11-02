"""Covariance between nearby pixels of a drizzled image.

Drizzling spreads the noise of each input pixel over the output pixels it
lands on, so neighboring output pixels are correlated.  With independent
input noise of variance ``var_k`` in exposure k, the weighted sums of
`Coadd` have the covariance ``sum_k W_k diag(var_k) W_k.T``, and the image,
those sums over the weights,

    Cov(img[o], img[o']) = sum_k (W_k diag(var_k) W_k.T)[o, o'] / (wsum[o] wsum[o'])

An input pixel reaches only the few output pixels under its footprint, so
the covariance is zero beyond a small offset between ``o`` and ``o'``.
`Covariance` keeps it for the offsets up to a radius in each coordinate,
one image per offset.  Entry ``(o, o')`` is the weighted dot product of
rows ``o`` and ``o'`` of ``W``, computed for every output pixel and offset
in parallel.  From it come the correlation of each pixel with its
neighbors and the noise factor of each pixel, by which the variance of a
sum over a large area exceeds the sum of its pixels' variances.

`NoiseFactor` gives the noise factor alone from two sums instead of one
per offset, in a second pass over the exposures once their weight sums
are known.
"""
import os
import warnings

import numba
import numpy as np

from .operator import _box_rows

__all__ = [
    "Covariance",
    "NoiseFactor",
]


class Covariance:
    """Covariance between nearby pixels of the image of drizzled exposures.

    Add the same operators as to `Coadd`, each with the variance of its
    exposure's data (for example the sum of its variance planes); the image
    is `Coadd.out_img`.  The covariance is kept for the offsets ``(dy, dx)``
    with ``|dy|, |dx| <= radius``, for half of them, listed in `offsets`:
    ``(0, 0)``, then ``(0, 1)`` to ``(0, radius)``, then ``(dy, -radius)``
    to ``(dy, radius)`` for ``dy`` from 1 to ``radius``.  The others follow
    from ``Cov(img[y, x], img[y - dy, x - dx]) = covariance(k)[y - dy, x - dx]``.

    Pixels farther apart than an input pixel's footprint share no input
    pixel and have no covariance; `Operator.reach` is that distance.  By
    default the radius is the reach of the first operator added.  An
    operator that reaches farther than the radius gives a warning, as the
    covariance beyond the radius is left out.

    The sums take ``2 * radius * (radius + 1) + 1`` output-sized images of
    ``dtype``; each exposure's contribution is computed in float64 before
    it is added.  Output pixels that no exposure reaches are not written,
    so the system does not give them memory until the results are made.
    The results are made one image at a time.

    Parameters
    ----------
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    radius : int, optional
        Largest offset kept, in output pixels, in y and in x; by default
        the reach of the first operator added.
    dtype : {numpy.float32, numpy.float64}
        Type of the sums and of the results.
    fillval : float
        Value where the pixel or its neighbor has no weight or variance, or
        the neighbor is outside the output grid.
    """

    def __init__(self, out_shape, radius=None, dtype=np.float32, fillval=np.nan):
        if radius is not None and radius < 0:
            raise ValueError(f"radius must not be negative, not {radius}")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"dtype must be float32 or float64, not {self.dtype}")
        self.out_shape = tuple(out_shape)
        self.fillval = fillval
        self.radius = None
        self.offsets = None
        self._cnum = None
        self._wsum = np.zeros(self.out_shape[0] * self.out_shape[1])
        if radius is not None:
            self._allocate(int(radius))

    def _allocate(self, radius):
        """Set the radius and offsets, and make the sums."""
        r = radius
        offsets = [(0, dx) for dx in range(r + 1)] + [(dy, dx) for dy in range(1, r + 1) for dx in range(-r, r + 1)]
        self.radius = r
        self.offsets = np.array(offsets, dtype=np.int64).reshape(-1, 2)
        self._cnum = np.zeros((len(self.offsets), self._wsum.size), dtype=self.dtype)

    def add(self, W, var, iscale=1.0):
        """Add one exposure.

        Parameters
        ----------
        W : Operator or scipy.sparse array
            The exposure's operator, as for `Coadd.add`.  A sparse array
            records no reach, so it needs the radius to be set.
        var : ndarray, shape (ny, nx)
            Variance of the exposure's data.
        iscale : float
            Factor applied to the data; its square is applied to ``var``.
        """
        var = np.asarray(var, dtype=np.float64)
        matrix, y0, x0, box_ny, box_nx = _box_rows(W, self.out_shape, var.size)
        reach = getattr(W, "reach", None)
        if self._cnum is None:
            if reach is None:
                raise ValueError("the radius must be given for operators that do not record their reach")
            self._allocate(reach)
        elif reach is not None and reach > self.radius:
            warnings.warn(f"the operator's input pixels reach {reach} output pixels, more than the radius {self.radius}; "
                          "the covariance at larger offsets is left out", stacklevel=2)
        if not matrix.has_sorted_indices:
            matrix = matrix.sorted_indices()
        _add_covariance(matrix.indptr, matrix.indices, matrix.data, y0, x0, box_ny, box_nx, self.out_shape[1],
                        var.ravel(), iscale * iscale, self.offsets, self._cnum, self._wsum)

    def covariance(self, k):
        """Covariance of each pixel with its neighbor at ``offsets[k]``.

        Parameters
        ----------
        k : int

        Returns
        -------
        ndarray, shape out_shape
            ``Cov(img[y, x], img[y + dy, x + dx])``; `fillval` where either
            pixel has no weight or the neighbor is off the grid.
        """
        self._check_added()
        dy, dx = self.offsets[k]
        out = np.empty(self.out_shape, dtype=self.dtype)
        _normalized(self._cnum[k], self._wsum, self.out_shape[0], self.out_shape[1], dy, dx, self.fillval, out)
        return out

    def correlation(self, k):
        """Correlation coefficient of each pixel with its neighbor at ``offsets[k]``.

        Parameters
        ----------
        k : int

        Returns
        -------
        ndarray, shape out_shape
            ``Cov(img[o], img[o']) / sqrt(Var(img[o]) Var(img[o']))``;
            `fillval` where either pixel has no variance or the neighbor is
            off the grid.
        """
        self._check_added()
        dy, dx = self.offsets[k]
        out = np.empty(self.out_shape, dtype=self.dtype)
        _correlated(self._cnum[k], self._cnum[0], self.out_shape[0], self.out_shape[1], dy, dx, self.fillval, out)
        return out

    @property
    def out_var(self):
        """ndarray, shape out_shape: variance of the image, `covariance` at offset (0, 0)."""
        return self.covariance(0)

    @property
    def out_cov(self):
        """ndarray, shape (n_offsets,) + out_shape: `covariance` for every offset, all in memory at once."""
        return np.array([self.covariance(k) for k in range(len(self.offsets))])

    @property
    def noise_factor(self):
        """ndarray, shape out_shape: covariance of each pixel with every pixel, over its variance.

        The variance of a sum over an area much larger than an input
        pixel's footprint is the sum of the variances of its pixels times
        this factor, so ``sqrt(var * noise_factor)`` is the uncertainty of
        each pixel for such sums, where ``sqrt(var)`` is that of the pixel
        alone.  For evenly dithered exposures it is the square of the
        noise ratio R of Casertano et al. (2000).  `fillval` where the
        pixel has no weight or variance.
        """
        self._check_added()
        out = np.empty(self.out_shape, dtype=self.dtype)
        _noise_factor(self._cnum, self._wsum, self.offsets, self.out_shape[0], self.out_shape[1], self.fillval, out)
        return out

    def write_correlation(self, path, overwrite=False):
        """Write the correlation of each pixel with its neighbors to a FITS file.

        Extension CORR is a float32 cube of shape ``(n_offsets - 1,) +
        out_shape`` whose plane ``k - 1`` is `correlation` for
        ``offsets[k]``, k from 1, and extension OFFSETS a table of their
        ``dy`` and ``dx``; the primary header has the radius.  The planes
        are made and written one at a time.  Needs astropy.

        Parameters
        ----------
        path : str or os.PathLike
        overwrite : bool
            Replace an existing file.
        """
        from astropy.io import fits

        # a string, as astropy's StreamingHDU takes only the name of a pathlib.Path
        path = os.fspath(path)
        self._check_added()
        if self.radius == 0:
            raise ValueError("radius 0 keeps no correlation to write")
        if os.path.exists(path):
            if not overwrite:
                raise OSError(f"{path} exists; pass overwrite=True to replace it")
            os.remove(path)

        primary = fits.PrimaryHDU()
        primary.header["RADIUS"] = (self.radius, "largest offset in y and x, output pixels")
        primary.writeto(path)
        ny, nx = self.out_shape
        header = fits.Header([
            ("XTENSION", "IMAGE"), ("BITPIX", -32), ("NAXIS", 3),
            ("NAXIS1", nx), ("NAXIS2", ny), ("NAXIS3", len(self.offsets) - 1),
            ("PCOUNT", 0), ("GCOUNT", 1), ("EXTNAME", "CORR"),
        ])
        stream = fits.StreamingHDU(path, header)
        try:
            for k in range(1, len(self.offsets)):
                stream.write(self.correlation(k).astype(np.float32))
        finally:
            stream.close()
        offsets = fits.BinTableHDU.from_columns([
            fits.Column(name="dy", format="J", array=self.offsets[1:, 0]),
            fits.Column(name="dx", format="J", array=self.offsets[1:, 1]),
        ], name="OFFSETS")
        with fits.open(path, mode="append") as hdus:
            hdus.append(offsets)

    def _check_added(self):
        if self._cnum is None:
            raise ValueError("no exposure has been added")


class NoiseFactor:
    """Noise factor of the image of drizzled exposures, in a second pass over them.

    The noise factor of `Covariance` needs ``2 r (r + 1) + 1`` output-sized
    sums, one per offset.  With the weight sums of all exposures known from
    a first pass, it needs two.  For ``C = sum_k W_k diag(var_k) W_k.T``,
    each exposure adds to

        T[o] = sum_o' C[o, o'] / wsum[o'] = W @ (var * (W.T @ (1 / wsum)))
        V[o] = C[o, o] = (W * W) @ var

    and ``noise_factor = wsum * T / V``: each pixel's covariance with every
    pixel, at every offset, over its variance.  No radius is needed.

    Add every exposure to a `Coadd`, then make a `NoiseFactor` from its
    weight sums and add the same operators again, kept or built again,
    each with the variance of its exposure.

    Parameters
    ----------
    wsum : ndarray, shape (ny_out, nx_out)
        Weight sums of all the exposures, such as `Coadd.out_wht`.  Kept,
        not copied, so it must not change while exposures are added.
    fillval : float
        Value where the pixel has no weight or variance.
    """

    def __init__(self, wsum, fillval=np.nan):
        wsum = np.asarray(wsum, dtype=np.float64)
        if wsum.ndim != 2:
            raise ValueError(f"wsum must be 2D, not of shape {wsum.shape}")
        self.out_shape = wsum.shape
        self.fillval = fillval
        self.n_images = 0
        self._wsum = wsum
        self._total = np.zeros(wsum.size)
        self._vnum = np.zeros(wsum.size)

    def add(self, W, var, iscale=1.0):
        """Add one exposure that is in the weight sums.

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
        wsum = self._wsum[y0:y0 + box_ny, x0:x0 + box_nx].ravel()
        reached = np.diff(matrix.indptr) > 0
        if np.any(reached & (wsum <= 0.0)):
            raise ValueError("the operator reaches output pixels without weight; was its exposure in the weight sums?")
        inverse = np.zeros(wsum.size)
        inverse[reached] = 1.0 / wsum[reached]
        # for each input pixel, its weights over the weight sums of the output pixels it reaches
        u = matrix.T @ inverse
        _add_noise_sums(matrix.indptr, matrix.indices, matrix.data, y0, x0, box_ny, box_nx, self.out_shape[1],
                        var.ravel(), iscale * iscale, u, self._total, self._vnum)
        self.n_images += 1

    @property
    def noise_factor(self):
        """ndarray, shape out_shape: covariance of each pixel with every pixel, over its variance.

        As `Covariance.noise_factor` with a radius of at least the reach;
        `fillval` where the pixel has no weight or variance.
        """
        if self.n_images == 0:
            raise ValueError("no exposure has been added")
        wsum = self._wsum.ravel()
        out = np.full(wsum.size, self.fillval, dtype=np.float64)
        ok = (wsum != 0.0) & (self._vnum > 0.0)
        out[ok] = wsum[ok] * self._total[ok] / self._vnum[ok]
        return out.reshape(self.out_shape)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _add_noise_sums(indptr, indices, values, y0, x0, box_ny, box_nx, nx_out, var, iscale2, u, total, vnum):
    """Add ``W @ (var * iscale2 * u)`` to total and ``(W * W) @ (var * iscale2)`` to vnum; rows without entries are left alone.

    Parameters
    ----------
    indptr, indices, values : ndarray
        CSR arrays of W, one row per pixel of the box.
    y0, x0, box_ny, box_nx : int
        First output row and column and shape of the box.
    nx_out : int
    var : ndarray, shape (n_in,)
    iscale2 : float
    u : ndarray, shape (n_in,)
        ``W.T @ (1 / wsum)``.
    total, vnum : ndarray of float64, shape (n_out,)
    """
    for by in numba.prange(box_ny):
        for bx in range(box_nx):
            row = by * box_nx + bx
            if indptr[row] == indptr[row + 1]:
                continue
            s = 0.0
            v = 0.0
            for k in range(indptr[row], indptr[row + 1]):
                w = values[k]
                vi = var[indices[k]] * iscale2
                s += w * vi * u[indices[k]]
                v += w * w * vi
            o = (y0 + by) * nx_out + x0 + bx
            total[o] += s
            vnum[o] += v


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
    cnum : ndarray, shape (n_offsets, n_out)
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


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _normalized(cnum, wsum, ny, nx, dy, dx, fillval, out):
    """``cnum[o] / (wsum[o] wsum[o'])`` for ``o' = o + (dy, dx)``, `fillval` where undefined.

    Parameters
    ----------
    cnum : ndarray, shape (ny * nx,)
        Summed covariance of one offset.
    wsum : ndarray of float64, shape (ny * nx,)
    ny, nx, dy, dx : int
    fillval : float
    out : ndarray, shape (ny, nx)
        Filled in place.
    """
    for y in numba.prange(ny):
        for x in range(nx):
            y2 = y + dy
            x2 = x + dx
            w = wsum[y * nx + x]
            if y2 >= ny or x2 < 0 or x2 >= nx or w == 0.0 or wsum[y2 * nx + x2] == 0.0:
                out[y, x] = fillval
            else:
                out[y, x] = cnum[y * nx + x] / (w * wsum[y2 * nx + x2])


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _correlated(cnum, cnum0, ny, nx, dy, dx, fillval, out):
    """``cnum[o] / sqrt(cnum0[o] cnum0[o'])`` for ``o' = o + (dy, dx)``; the weights cancel.

    Parameters
    ----------
    cnum, cnum0 : ndarray, shape (ny * nx,)
        Summed covariance of one offset and of offset (0, 0).
    ny, nx, dy, dx : int
    fillval : float
    out : ndarray, shape (ny, nx)
        Filled in place.
    """
    for y in numba.prange(ny):
        for x in range(nx):
            y2 = y + dy
            x2 = x + dx
            v = cnum0[y * nx + x]
            if y2 >= ny or x2 < 0 or x2 >= nx or v <= 0.0 or cnum0[y2 * nx + x2] <= 0.0:
                out[y, x] = fillval
            else:
                out[y, x] = cnum[y * nx + x] / np.sqrt(np.float64(v) * cnum0[y2 * nx + x2])


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _noise_factor(cnum, wsum, offsets, ny, nx, fillval, out):
    """Sum of the covariance of each pixel with all pixels, over its variance.

    With ``Cov(o, o + d) = cnum[k, o] / (wsum[o] wsum[o + d])`` for the kept
    offsets d and ``Cov(o, o - d) = cnum[k, o - d] / (wsum[o - d] wsum[o])``
    for the others, and ``Var(o) = cnum[0, o] / wsum[o]**2``, the factor is
    ``wsum[o] / cnum[0, o]`` times the sum of ``cnum[k, o] / wsum[o + d]``
    and ``cnum[k, o - d] / wsum[o - d]`` over the neighbors with weight.

    Parameters
    ----------
    cnum : ndarray, shape (n_offsets, ny * nx)
    wsum : ndarray of float64, shape (ny * nx,)
    offsets : ndarray of int64, shape (n_offsets, 2)
    ny, nx : int
    fillval : float
    out : ndarray, shape (ny, nx)
        Filled in place.
    """
    for y in numba.prange(ny):
        for x in range(nx):
            o = y * nx + x
            w = wsum[o]
            v = cnum[0, o]
            if w == 0.0 or v <= 0.0:
                out[y, x] = fillval
                continue
            total = 0.0
            for m in range(offsets.shape[0]):
                dy = offsets[m, 0]
                dx = offsets[m, 1]
                y2 = y + dy
                x2 = x + dx
                if y2 < ny and 0 <= x2 < nx and wsum[y2 * nx + x2] != 0.0:
                    total += cnum[m, o] / wsum[y2 * nx + x2]
                y3 = y - dy
                x3 = x - dx
                if m > 0 and y3 >= 0 and 0 <= x3 < nx and wsum[y3 * nx + x3] != 0.0:
                    total += cnum[m, y3 * nx + x3] / wsum[y3 * nx + x3]
            out[y, x] = total * w / v
