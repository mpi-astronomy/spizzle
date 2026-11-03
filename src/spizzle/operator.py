"""The drizzle operator and the co-addition of exposures with it.

Drizzling one exposure is the linear map ``W @ data``, with
``W[o, i] = area[o, i] * weight[i] * weight_scale / jaco[i]``, the overlap
area of input pixel ``i`` with output pixel ``o`` times the input weight,
divided by the area of the input pixel on the output grid.  ``W`` depends
on the pixel map, ``pixfrac`` and the weights, not on the data, so it is
built once per exposure and applied to the science data, the weights and
every variance plane:

    num  += W @ data
    wsum += W @ 1
    vnum += (W * W) @ var

and the output, ``num / wsum`` and ``vnum / wsum**2``, is formed once at
the end in float64.  In exact arithmetic this is the running weighted mean
that drizzle updates for every contribution.  ``W`` is built and applied
on numba's threads, and the results do not depend on their number.

An exposure usually lands on a small part of the output grid, which can be
a large mosaic, so `Operator` keeps ``W`` to the rows of the exposure's
bounding box on the output grid.
"""
from dataclasses import dataclass

import numba
import numpy as np
from scipy import sparse

from .geometry import corners, jacobian, overlap_triplets

__all__ = [
    "build_operator",
    "drizzle_operator",
    "Operator",
    "Coadd",
]


@dataclass(frozen=True, eq=False)
class Operator:
    """Drizzle operator of one exposure on its bounding box of the output grid.

    ``matrix @ data.ravel()`` gives the exposure's weighted sums on the
    output pixels of ``box``; the exposure contributes nothing outside it.
    Keeping to the box makes the operator, and the work to build and apply
    it, scale with the exposure rather than with the output grid.

    Parameters
    ----------
    matrix : scipy.sparse.csr_array, shape (box_ny * box_nx, ny * nx)
        Row ``r`` is output pixel ``(y0 + r // box_nx, x0 + r % box_nx)``.
    box : tuple of slice
        ``(slice(y0, y1), slice(x0, x1))``, the smallest box of output
        pixels that holds every entry, so that ``image[box]`` is the
        exposure's region of an output image; empty if the exposure misses
        the output grid.
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    reach : int or None
        Largest distance in y or in x, in output pixels, between two output
        pixels that share an input pixel: drizzled noise is uncorrelated
        beyond it (see `Covariance`).  None if not known.
    """

    matrix: sparse.csr_array
    box: tuple
    out_shape: tuple
    reach: int = None

    @property
    def box_shape(self):
        """tuple of int: ``(box_ny, box_nx)``, the shape of ``image[box]``."""
        return (self.box[0].stop - self.box[0].start, self.box[1].stop - self.box[1].start)

    def full(self):
        """The operator on the whole output grid.

        Returns
        -------
        scipy.sparse.csr_array, shape (ny_out * nx_out, ny * nx)
            Shares its indices and values with `matrix`.
        """
        counts = np.zeros(self.out_shape, dtype=np.int64)
        counts[self.box] = np.diff(self.matrix.indptr).reshape(self.box_shape)
        indptr = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts.ravel(), out=indptr[1:])
        return sparse.csr_array((self.matrix.data, self.matrix.indices, indptr), shape=(counts.size, self.matrix.shape[1]))


def build_operator(triplets, jaco, out_shape, weights=None, weight_scale=1.0):
    """Drizzle operator of one exposure from its overlap geometry and weights.

    Entries whose weight is zero are left out, as drizzle skips them, so a
    pixel with zero weight contributes nothing even where its data are NaN.

    Parameters
    ----------
    triplets : tuple of ndarray
        ``(in_idx, out_idx, area)`` from `overlap_triplets`.
    jaco : ndarray, shape (ny, nx)
        From `jacobian`.
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    weights : ndarray, shape (ny, nx), optional
        Input pixel weights; all 1 if not given.
    weight_scale : float
        Factor applied to the weights, drizzle's ``wht_scale``.

    Returns
    -------
    Operator
    """
    in_idx, out_idx, area = triplets
    jaco = np.asarray(jaco, dtype=np.float64)
    if weights is None:
        w = weight_scale / jaco
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != jaco.shape:
            raise ValueError(f"weights shape {weights.shape} does not match the input shape {jaco.shape}")
        w = weights * weight_scale / jaco
    w = w.ravel()
    ny_out, nx_out = out_shape

    # sort the triplets by output row, a chunk of them per thread, then by
    # output column within each row; both sorts keep the triplet order, so
    # the entries of each output pixel stay in input pixel order
    n_chunks = max(1, min(4 * numba.get_num_threads(), in_idx.size))
    chunk_starts = np.linspace(0, in_idx.size, n_chunks + 1).astype(np.int64)
    # start each chunk at the first triplet of an input pixel, so that every
    # input pixel's footprint is in one chunk
    chunk_starts[1:-1] = np.searchsorted(in_idx, in_idx[chunk_starts[1:-1]], side="left")
    counts = np.zeros((n_chunks, ny_out), dtype=np.int64)
    col_min = np.empty(n_chunks, dtype=np.int64)
    col_max = np.empty(n_chunks, dtype=np.int64)
    reach = np.zeros(n_chunks, dtype=np.int64)
    _count_rows(in_idx, out_idx, area, w, nx_out, chunk_starts, counts, col_min, col_max, reach)
    rows = np.flatnonzero(counts.sum(axis=0))
    if rows.size:
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(col_min.min()), int(col_max.max()) + 1
    else:
        y0 = y1 = x0 = x1 = 0
    box_ny, box_nx = y1 - y0, x1 - x0

    counts = counts[:, y0:y1]
    row_starts = np.zeros(box_ny + 1, dtype=np.int64)
    np.cumsum(counts.sum(axis=0), out=row_starts[1:])
    offsets = row_starts[:-1] + np.cumsum(counts, axis=0) - counts
    nnz = int(row_starts[-1])
    row_cols = np.empty(nnz, dtype=np.int64)
    row_ins = np.empty(nnz, dtype=in_idx.dtype)
    row_values = np.empty(nnz)
    _bucket_rows(in_idx, out_idx, area, w, nx_out, y0, x0, chunk_starts, offsets, row_cols, row_ins, row_values)
    indptr = np.empty(box_ny * box_nx + 1, dtype=np.int64)
    indptr[0] = 0
    indices = np.empty(nnz, dtype=in_idx.dtype)
    data = np.empty(nnz)
    _sort_rows(row_starts, box_nx, row_cols, row_ins, row_values, indptr, indices, data)
    matrix = sparse.csr_array((data, indices, indptr), shape=(box_ny * box_nx, jaco.size))
    # each output pixel's entries are in input pixel order, so scipy need not check
    matrix.has_sorted_indices = True
    return Operator(matrix, (slice(y0, y1), slice(x0, x1)), (ny_out, nx_out), int(reach.max()))


def drizzle_operator(pixmap, out_shape, pixfrac=1.0, weights=None, weight_scale=1.0, region=None):
    """Drizzle operator of one exposure, square kernel.

    `corners`, `jacobian`, `overlap_triplets` and `build_operator` in turn.
    Call those directly to keep the geometry and re-weight it.

    With ``region``, only the input pixels in it are drizzled, as with
    drizzle's ``xmin``, ``xmax``, ``ymin`` and ``ymax``; the others are not
    visited, which saves their overlap geometry.  Their corners still come
    from the whole pixel map, so the region's edge pixels are the same as
    without it.  To leave out other pixels, give them zero weight: entries
    of zero weight are left out of the operator, so their data are never
    used.

    Parameters
    ----------
    pixmap : ndarray, shape (ny, nx, 2)
        Output ``x`` and ``y`` coordinates of each input pixel center.
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    pixfrac : float
        Linear size of the shrunken pixel as a fraction of the input pixel.
    weights : ndarray, shape (ny, nx), optional
        Input pixel weights; all 1 if not given.
    weight_scale : float
        Factor applied to the weights, drizzle's ``wht_scale``.
    region : tuple of slice, optional
        ``(slice(y0, y1), slice(x0, x1))`` of the input pixels to drizzle;
        all by default.  See `overlap_triplets`.

    Returns
    -------
    Operator
    """
    c = corners(pixmap, pixfrac)
    return build_operator(overlap_triplets(c, out_shape, region), jacobian(c), out_shape, weights, weight_scale)


class Coadd:
    """Weighted mean of drizzled exposures on one output grid, in float64.

    Parameters
    ----------
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    n_var : int
        Number of variance planes given with each exposure.
    fillval : float
        Value of output pixels without weight, in `out_img` and `out_var`.
    context : bool
        Keep drizzle's context, one int32 plane of the output size per 32
        exposures, as `out_ctx`.
    """

    def __init__(self, out_shape, n_var=0, fillval=np.nan, context=False):
        self.out_shape = tuple(out_shape)
        self.fillval = fillval
        self.context = context
        self.n_images = 0
        n_out = self.out_shape[0] * self.out_shape[1]
        self._num = np.zeros(n_out)
        self._wsum = np.zeros(n_out)
        self._vnum = [np.zeros(n_out) for _ in range(n_var)]
        self._ctx = []

    def add(self, W, data, variances=(), iscale=1.0):
        """Add one exposure.

        Parameters
        ----------
        W : Operator or scipy.sparse array
            The exposure's operator, from `drizzle_operator` or
            `build_operator`, or a sparse array of shape
            (ny_out * nx_out, ny * nx) on the whole output grid.
        data : ndarray, shape (ny, nx)
            Science data.
        variances : sequence of ndarray or None
            One variance plane per ``n_var``, each of shape (ny, nx); None
            adds nothing to that plane.
        iscale : float
            Factor applied to the data, and its square to the variances.
        """
        data = np.asarray(data, dtype=np.float64)
        matrix, y0, x0, box_ny, box_nx = _box_rows(W, self.out_shape, data.size)
        if len(variances) != len(self._vnum):
            raise ValueError(f"expected {len(self._vnum)} variance planes, got {len(variances)}")

        if self.context:
            plane, bit = divmod(self.n_images, 32)
            if plane == len(self._ctx):
                self._ctx.append(np.zeros(self._num.size, dtype=np.int32))
            ctx, mask = self._ctx[plane], np.uint32(1 << bit).view(np.int32)
        else:
            ctx, mask = np.zeros(1, dtype=np.int32), np.int32(0)
        rows = (matrix.indptr, matrix.indices, matrix.data, y0, x0, box_ny, box_nx, self.out_shape[1])
        _add_data(*rows, data.ravel(), iscale, self._num, self._wsum, ctx, mask)
        for vnum, var in zip(self._vnum, variances):
            if var is not None:
                _add_variance(*rows, np.asarray(var, dtype=np.float64).ravel(), iscale * iscale, vnum)
        self.n_images += 1

    @property
    def out_img(self):
        """ndarray, shape out_shape: weighted mean of the data, `fillval` without weight."""
        return self._normalize(self._num, self._wsum)

    @property
    def out_wht(self):
        """ndarray, shape out_shape: sum of the weights."""
        return self._wsum.reshape(self.out_shape)

    @property
    def out_var(self):
        """list of ndarray, shape out_shape: variance of `out_img` from each variance plane."""
        wsum2 = self._wsum * self._wsum
        return [self._normalize(vnum, wsum2) for vnum in self._vnum]

    @property
    def out_ctx(self):
        """ndarray of int32, shape (n_planes,) + out_shape: bit ``k % 32`` of plane ``k // 32`` set where exposure k lands.

        None unless the Coadd was made with ``context=True``.
        """
        if not self.context:
            return None
        return np.array(self._ctx, dtype=np.int32).reshape((len(self._ctx),) + self.out_shape)

    def _normalize(self, num, den):
        """``num / den`` where the weight sum is non-zero, `fillval` elsewhere."""
        out = np.full(num.size, self.fillval, dtype=np.float64)
        nonzero = self._wsum != 0.0
        out[nonzero] = num[nonzero] / den[nonzero]
        return out.reshape(self.out_shape)


def _box_rows(W, out_shape, n_in):
    """CSR rows of an operator and the box of the output grid that they cover.

    Parameters
    ----------
    W : Operator or scipy.sparse array
        As for `Coadd.add`.
    out_shape : tuple of int
        ``(ny_out, nx_out)`` of the grid it is added to.
    n_in : int
        Number of input pixels it is applied to.

    Returns
    -------
    matrix : scipy.sparse.csr_array
        One row per pixel of the box.
    y0, x0, box_ny, box_nx : int
        First output row and column and shape of the box.
    """
    if isinstance(W, Operator):
        if tuple(W.out_shape) != tuple(out_shape):
            raise ValueError(f"operator for output shape {tuple(W.out_shape)} does not match output shape {tuple(out_shape)}")
        matrix = sparse.csr_array(W.matrix)
        y0, x0 = W.box[0].start, W.box[1].start
        box_ny, box_nx = W.box_shape
    else:
        matrix = sparse.csr_array(W)
        y0, x0 = 0, 0
        box_ny, box_nx = out_shape
    if matrix.shape != (box_ny * box_nx, n_in):
        raise ValueError(f"operator shape {matrix.shape} does not match {box_ny * box_nx} output pixels and {n_in} data")
    return matrix, y0, x0, box_ny, box_nx


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _count_rows(in_idx, out_idx, area, w, nx_out, chunk_starts, counts, col_min, col_max, reach):
    """Count the non-zero entries of each chunk of triplets in each output row, and find their column range and reach.

    Parameters
    ----------
    in_idx, out_idx, area : ndarray, shape (n,)
        Triplets from `overlap_triplets`.
    w : ndarray, shape (n_in,)
        Factor of each input pixel: weight times weight scale over jacobian.
    nx_out : int
    chunk_starts : ndarray of int64, shape (n_chunks + 1,)
        Triplet ranges of the chunks.
    counts : ndarray of int64, shape (n_chunks, ny_out)
        Incremented in place.
    col_min, col_max : ndarray of int64, shape (n_chunks,)
        Set to the smallest and largest output column of each chunk's
        non-zero entries; ``nx_out`` and -1 for a chunk without any.
    reach : ndarray of int64, shape (n_chunks,)
        Set to the largest extent, rows or columns minus one, of the
        non-zero entries of one input pixel in the chunk.  Each input
        pixel's triplets must be consecutive and in one chunk.
    """
    for c in numba.prange(chunk_starts.size - 1):
        lo = nx_out
        hi = -1
        far = 0
        pixel = -1
        rlo = rhi = clo = chi = 0
        for k in range(chunk_starts[c], chunk_starts[c + 1]):
            if area[k] * w[in_idx[k]] != 0.0:
                row = out_idx[k] // nx_out
                col = out_idx[k] - row * nx_out
                counts[c, row] += 1
                lo = min(lo, col)
                hi = max(hi, col)
                if in_idx[k] != pixel:
                    far = max(far, rhi - rlo, chi - clo)
                    pixel = in_idx[k]
                    rlo = rhi = row
                    clo = chi = col
                else:
                    rlo = min(rlo, row)
                    rhi = max(rhi, row)
                    clo = min(clo, col)
                    chi = max(chi, col)
        col_min[c] = lo
        col_max[c] = hi
        reach[c] = max(far, rhi - rlo, chi - clo)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _bucket_rows(in_idx, out_idx, area, w, nx_out, y0, x0, chunk_starts, offsets, row_cols, row_ins, row_values):
    """Write the non-zero entries grouped by row of the box, in triplet order within each row.

    Parameters
    ----------
    in_idx, out_idx, area, w, nx_out, chunk_starts
        As for `_count_rows`.
    y0, x0 : int
        First output row and column of the box.
    offsets : ndarray of int64, shape (n_chunks, box_ny)
        First slot of each chunk in each row of the box.
    row_cols, row_ins, row_values : ndarray, shape (nnz,)
        Column in the box, input index and value of each entry, filled in
        place.
    """
    for c in numba.prange(chunk_starts.size - 1):
        cursor = offsets[c].copy()
        for k in range(chunk_starts[c], chunk_starts[c + 1]):
            value = area[k] * w[in_idx[k]]
            if value != 0.0:
                out_row = out_idx[k] // nx_out
                row = out_row - y0
                slot = cursor[row]
                cursor[row] += 1
                row_cols[slot] = out_idx[k] - out_row * nx_out - x0
                row_ins[slot] = in_idx[k]
                row_values[slot] = value


@numba.njit(parallel=True, cache=True)
def _sort_rows(row_starts, box_nx, row_cols, row_ins, row_values, indptr, indices, data):
    """Sort the entries of each row of the box by column, keeping their order within a column.

    Parameters
    ----------
    row_starts : ndarray of int64, shape (box_ny + 1,)
        Entry range of each row of the box.
    box_nx : int
    row_cols, row_ins, row_values : ndarray, shape (nnz,)
        From `_bucket_rows`.
    indptr, indices, data : ndarray
        The CSR arrays, filled in place except ``indptr[0]``.
    """
    for row in numba.prange(row_starts.size - 1):
        lo = row_starts[row]
        hi = row_starts[row + 1]
        # starts[col] is the first slot of column col, relative to lo
        starts = np.zeros(box_nx + 1, dtype=np.int64)
        for slot in range(lo, hi):
            starts[row_cols[slot] + 1] += 1
        for col in range(box_nx):
            starts[col + 1] += starts[col]
            indptr[row * box_nx + col + 1] = lo + starts[col + 1]
        for slot in range(lo, hi):
            dst = lo + starts[row_cols[slot]]
            starts[row_cols[slot]] += 1
            indices[dst] = row_ins[slot]
            data[dst] = row_values[slot]


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _add_data(indptr, indices, values, y0, x0, box_ny, box_nx, nx_out, data, iscale, num, wsum, ctx, mask):
    """Add ``W @ (data * iscale)`` to num and ``W @ 1`` to wsum, and set ``mask`` in ctx where a row of W has a positive entry.

    drizzle sets the context bit for any positive contribution.  Rows
    without entries are left alone.  With ``mask`` zero, ctx is not
    touched and can be any int32 array.

    Parameters
    ----------
    indptr, indices, values : ndarray
        CSR arrays of W, one row per pixel of the box.
    y0, x0, box_ny, box_nx : int
        First output row and column and shape of the box.
    nx_out : int
    data : ndarray, shape (n_in,)
    iscale : float
    num, wsum : ndarray of float64, shape (n_out,)
    ctx : ndarray of int32, shape (n_out,)
    mask : int32
    """
    for by in numba.prange(box_ny):
        for bx in range(box_nx):
            row = by * box_nx + bx
            if indptr[row] == indptr[row + 1]:
                continue
            s = 0.0
            sw = 0.0
            hit = False
            for k in range(indptr[row], indptr[row + 1]):
                w = values[k]
                s += w * (data[indices[k]] * iscale)
                sw += w
                hit |= w > 0.0
            o = (y0 + by) * nx_out + x0 + bx
            num[o] += s
            wsum[o] += sw
            if hit and mask != 0:
                ctx[o] |= mask


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _add_variance(indptr, indices, values, y0, x0, box_ny, box_nx, nx_out, var, iscale2, vnum):
    """Add ``(W * W) @ (var * iscale2)`` to vnum; rows without entries are left alone.

    Parameters
    ----------
    indptr, indices, values, y0, x0, box_ny, box_nx, nx_out
        As for `_add_data`.
    var : ndarray, shape (n_in,)
    iscale2 : float
    vnum : ndarray of float64, shape (n_out,)
    """
    for by in numba.prange(box_ny):
        for bx in range(box_nx):
            row = by * box_nx + bx
            if indptr[row] == indptr[row + 1]:
                continue
            s = 0.0
            for k in range(indptr[row], indptr[row + 1]):
                w = values[k]
                s += w * w * (var[indices[k]] * iscale2)
            vnum[(y0 + by) * nx_out + x0 + bx] += s
