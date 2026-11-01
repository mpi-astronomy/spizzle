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
"""
import numba
import numpy as np
from scipy import sparse

from .geometry import corners, jacobian, overlap_triplets

__all__ = [
    "build_operator",
    "drizzle_operator",
    "Coadd",
]


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
    scipy.sparse.csr_array, shape (ny_out * nx_out, ny * nx)
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
    counts = np.zeros((n_chunks, ny_out), dtype=np.int64)
    _count_rows(in_idx, out_idx, area, w, nx_out, chunk_starts, counts)
    row_starts = np.zeros(ny_out + 1, dtype=np.int64)
    np.cumsum(counts.sum(axis=0), out=row_starts[1:])
    offsets = row_starts[:-1] + np.cumsum(counts, axis=0) - counts
    nnz = int(row_starts[-1])
    row_cols = np.empty(nnz, dtype=np.int64)
    row_ins = np.empty(nnz, dtype=in_idx.dtype)
    row_values = np.empty(nnz)
    _bucket_rows(in_idx, out_idx, area, w, nx_out, chunk_starts, offsets, row_cols, row_ins, row_values)
    indptr = np.empty(ny_out * nx_out + 1, dtype=np.int64)
    indptr[0] = 0
    indices = np.empty(nnz, dtype=in_idx.dtype)
    data = np.empty(nnz)
    _sort_rows(row_starts, nx_out, row_cols, row_ins, row_values, indptr, indices, data)
    return sparse.csr_array((data, indices, indptr), shape=(ny_out * nx_out, jaco.size))


def drizzle_operator(pixmap, out_shape, pixfrac=1.0, weights=None, weight_scale=1.0):
    """Drizzle operator of one exposure, square kernel.

    `corners`, `jacobian`, `overlap_triplets` and `build_operator` in turn.
    Call those directly to keep the geometry and re-weight it.

    Parameters
    ----------
    pixmap : ndarray, shape (ny, nx, 2)
        Output ``x`` and ``y`` coordinates of each input pixel centre.
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    pixfrac : float
        Linear size of the shrunken pixel as a fraction of the input pixel.
    weights : ndarray, shape (ny, nx), optional
        Input pixel weights; all 1 if not given.
    weight_scale : float
        Factor applied to the weights, drizzle's ``wht_scale``.

    Returns
    -------
    scipy.sparse.csr_array, shape (ny_out * nx_out, ny * nx)
    """
    c = corners(pixmap, pixfrac)
    return build_operator(overlap_triplets(c, out_shape), jacobian(c), out_shape, weights, weight_scale)


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
    """

    def __init__(self, out_shape, n_var=0, fillval=np.nan):
        self.out_shape = tuple(out_shape)
        self.fillval = fillval
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
        W : scipy.sparse array, shape (ny_out * nx_out, ny * nx)
            The exposure's operator, from `drizzle_operator` or
            `build_operator`.
        data : ndarray, shape (ny, nx)
            Science data.
        variances : sequence of ndarray or None
            One variance plane per ``n_var``, each of shape (ny, nx); None
            adds nothing to that plane.
        iscale : float
            Factor applied to the data, and its square to the variances.
        """
        W = sparse.csr_array(W)
        data = np.asarray(data, dtype=np.float64)
        if W.shape != (self._num.size, data.size):
            raise ValueError(f"operator shape {W.shape} does not match output size {self._num.size} and data size {data.size}")
        if len(variances) != len(self._vnum):
            raise ValueError(f"expected {len(self._vnum)} variance planes, got {len(variances)}")

        plane, bit = divmod(self.n_images, 32)
        if plane == len(self._ctx):
            self._ctx.append(np.zeros(self._num.size, dtype=np.int32))
        mask = np.uint32(1 << bit).view(np.int32)
        _add_data(W.indptr, W.indices, W.data, data.ravel(), iscale, self._num, self._wsum, self._ctx[plane], mask)
        for vnum, var in zip(self._vnum, variances):
            if var is not None:
                var = np.asarray(var, dtype=np.float64).ravel()
                _add_variance(W.indptr, W.indices, W.data, var, iscale * iscale, vnum)
        self.n_images += 1

    @property
    def out_img(self):
        """ndarray, shape out_shape: weighted mean of the data, `fillval` without weight."""
        return self._normalise(self._num, self._wsum)

    @property
    def out_wht(self):
        """ndarray, shape out_shape: sum of the weights."""
        return self._wsum.reshape(self.out_shape)

    @property
    def out_var(self):
        """list of ndarray, shape out_shape: variance of `out_img` from each variance plane."""
        wsum2 = self._wsum * self._wsum
        return [self._normalise(vnum, wsum2) for vnum in self._vnum]

    @property
    def out_ctx(self):
        """ndarray of int32, shape (n_planes,) + out_shape: bit ``k % 32`` of plane ``k // 32`` set where exposure k lands."""
        return np.array(self._ctx, dtype=np.int32).reshape((len(self._ctx),) + self.out_shape)

    def _normalise(self, num, den):
        """``num / den`` where the weight sum is non-zero, `fillval` elsewhere."""
        out = np.full(num.size, self.fillval, dtype=np.float64)
        nonzero = self._wsum != 0.0
        out[nonzero] = num[nonzero] / den[nonzero]
        return out.reshape(self.out_shape)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _count_rows(in_idx, out_idx, area, w, nx_out, chunk_starts, counts):
    """Count the non-zero entries of each chunk of triplets in each output row.

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
    """
    for c in numba.prange(chunk_starts.size - 1):
        for k in range(chunk_starts[c], chunk_starts[c + 1]):
            if area[k] * w[in_idx[k]] != 0.0:
                counts[c, out_idx[k] // nx_out] += 1


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _bucket_rows(in_idx, out_idx, area, w, nx_out, chunk_starts, offsets, row_cols, row_ins, row_values):
    """Write the non-zero entries grouped by output row, in triplet order within each row.

    Parameters
    ----------
    in_idx, out_idx, area, w, nx_out, chunk_starts
        As for `_count_rows`.
    offsets : ndarray of int64, shape (n_chunks, ny_out)
        First slot of each chunk in each output row.
    row_cols, row_ins, row_values : ndarray, shape (nnz,)
        Output column, input index and value of each entry, filled in place.
    """
    for c in numba.prange(chunk_starts.size - 1):
        cursor = offsets[c].copy()
        for k in range(chunk_starts[c], chunk_starts[c + 1]):
            value = area[k] * w[in_idx[k]]
            if value != 0.0:
                row = out_idx[k] // nx_out
                slot = cursor[row]
                cursor[row] += 1
                row_cols[slot] = out_idx[k] - row * nx_out
                row_ins[slot] = in_idx[k]
                row_values[slot] = value


@numba.njit(parallel=True, cache=True)
def _sort_rows(row_starts, nx_out, row_cols, row_ins, row_values, indptr, indices, data):
    """Sort the entries of each output row by column, keeping their order within a column.

    Parameters
    ----------
    row_starts : ndarray of int64, shape (ny_out + 1,)
        Entry range of each output row.
    nx_out : int
    row_cols, row_ins, row_values : ndarray, shape (nnz,)
        From `_bucket_rows`.
    indptr, indices, data : ndarray
        The CSR arrays, filled in place except ``indptr[0]``.
    """
    for row in numba.prange(row_starts.size - 1):
        lo = row_starts[row]
        hi = row_starts[row + 1]
        # starts[col] is the first slot of column col, relative to lo
        starts = np.zeros(nx_out + 1, dtype=np.int64)
        for slot in range(lo, hi):
            starts[row_cols[slot] + 1] += 1
        for col in range(nx_out):
            starts[col + 1] += starts[col]
            indptr[row * nx_out + col + 1] = lo + starts[col + 1]
        for slot in range(lo, hi):
            dst = lo + starts[row_cols[slot]]
            starts[row_cols[slot]] += 1
            indices[dst] = row_ins[slot]
            data[dst] = row_values[slot]


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _add_data(indptr, indices, values, data, iscale, num, wsum, ctx, mask):
    """Add ``W @ (data * iscale)`` to num and ``W @ 1`` to wsum, and set ``mask`` in ctx where a row of W has a positive entry.

    drizzle sets the context bit for any positive contribution.  Rows
    without entries are left alone.

    Parameters
    ----------
    indptr, indices, values : ndarray
        CSR arrays of W.
    data : ndarray, shape (n_in,)
    iscale : float
    num, wsum : ndarray of float64, shape (n_out,)
    ctx : ndarray of int32, shape (n_out,)
    mask : int32
    """
    for row in numba.prange(indptr.size - 1):
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
        num[row] += s
        wsum[row] += sw
        if hit:
            ctx[row] |= mask


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _add_variance(indptr, indices, values, var, iscale2, vnum):
    """Add ``(W * W) @ (var * iscale2)`` to vnum; rows without entries are left alone.

    Parameters
    ----------
    indptr, indices, values : ndarray
        CSR arrays of W.
    var : ndarray, shape (n_in,)
    iscale2 : float
    vnum : ndarray of float64, shape (n_out,)
    """
    for row in numba.prange(indptr.size - 1):
        if indptr[row] == indptr[row + 1]:
            continue
        s = 0.0
        for k in range(indptr[row], indptr[row + 1]):
            w = values[k]
            s += w * w * (var[indices[k]] * iscale2)
        vnum[row] += s
