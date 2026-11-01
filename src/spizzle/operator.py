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
that drizzle updates for every contribution.
"""
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
    values = area * w.ravel()[in_idx]
    keep = values != 0.0
    n_out = out_shape[0] * out_shape[1]
    return sparse.csr_array((values[keep], (out_idx[keep], in_idx[keep])), shape=(n_out, jaco.size))


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

        self._num += W @ (data.ravel() * iscale)
        self._wsum += W @ np.ones(data.size)
        if self._vnum:
            W2 = W.multiply(W)
            iscale2 = iscale * iscale
            for vnum, var in zip(self._vnum, variances):
                if var is not None:
                    vnum += W2 @ (np.asarray(var, dtype=np.float64).ravel() * iscale2)

        # drizzle sets an output pixel's context bit for any positive contribution
        plane, bit = divmod(self.n_images, 32)
        if plane == len(self._ctx):
            self._ctx.append(np.zeros(self._num.size, dtype=np.int32))
        positive = sparse.csr_array(((W.data > 0).astype(np.float32), W.indices, W.indptr), shape=W.shape)
        hit = positive @ np.ones(W.shape[1], dtype=np.float32) > 0
        self._ctx[plane][hit] |= np.uint32(1 << bit).view(np.int32)
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
