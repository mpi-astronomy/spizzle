"""The variance planes of a mosaic as the JWST pipeline computes them.

stcal's ``Resample`` does not propagate the variance through the drizzle
weights.  For each exposure and variance plane it drizzles the square root
of the variance as an image, a weighted mean of the errors with the
science weights, and squares the result.  It then combines the exposures
with one weight per output pixel and exposure: one over the resampled
read-noise variance, or the exposure time.  The exact variance of the
image is `Coadd.out_var`; `StcalVariance` reproduces stcal's arithmetic
from the same operators, to check spizzle against the pipeline.
"""
import numba
import numpy as np

from .operator import _box_rows

__all__ = [
    "StcalVariance",
]


class StcalVariance:
    """Variance planes and error of a mosaic as stcal 1.20 resamples them, for imaging.

    For each exposure and variance plane q, the resampled error is
    ``e_q = iscale * (W @ sqrt(var_q)) / (W @ 1)``, drizzle's weighted mean
    of ``sqrt(var_q)``, and each output pixel's weight for the exposure is
    ``w = 1 / e_r**2`` for the read-noise plane r ("ivm", 0 where ``e_r``
    is 0 or not finite) or the exposure time ("exptime").  Then

        var_q = sum(e_q**2 * w**2) / sum(w)**2 / pixel_scale_ratio**2

    over the exposures where ``e_q`` is finite and ``w > 0``: stcal's
    ``resample_variance_arrays`` and ``finalize_resample_variance``.  ERR is
    ``sqrt`` of the sum of the planes.

    Parameters
    ----------
    out_shape : tuple of int
        ``(ny_out, nx_out)``.
    n_var : int
        Number of variance planes given with each exposure.
    rnoise : int
        Index of the read-noise plane, for "ivm" weights; 0 in stcal's
        order of ``var_rnoise``, ``var_flat``, ``var_poisson``.
    weight_type : {"ivm", "exptime"}
        stcal's weighting of the exposures.
    pixel_scale_ratio : float
        Output over input pixel scale; the variance is divided by its square.
    fillval : float
        Value of output pixels where no exposure adds to a plane.
    """

    def __init__(self, out_shape, n_var=3, rnoise=0, weight_type="ivm", pixel_scale_ratio=1.0, fillval=np.nan):
        if weight_type not in ("ivm", "exptime"):
            raise ValueError(f"weight_type must be 'ivm' or 'exptime', not {weight_type!r}")
        if not 0 <= rnoise < n_var:
            raise ValueError(f"rnoise must index one of the {n_var} variance planes, not {rnoise}")
        self.out_shape = tuple(out_shape)
        self.rnoise = rnoise
        self.weight_type = weight_type
        self.pixel_scale_ratio = pixel_scale_ratio
        self.fillval = fillval
        self.n_images = 0
        n_out = self.out_shape[0] * self.out_shape[1]
        self._wsum = np.zeros((n_var, n_out))
        self._wt = np.zeros((n_var, n_out))

    def add(self, W, variances, iscale=1.0, exposure_time=None):
        """Add one exposure.

        Parameters
        ----------
        W : Operator or scipy.sparse array
            The exposure's operator, as for `Coadd.add`.
        variances : sequence of ndarray
            One variance plane per ``n_var``, each of shape (ny, nx).  For
            "ivm" weights the read-noise plane is needed; other planes may
            be None, which adds nothing to them, as stcal skips missing
            planes.
        iscale : float
            Factor applied to the resampled errors.
        exposure_time : float, optional
            The exposure's weight for "exptime": stcal's measurement time,
            or the exposure time.
        """
        n_var = self._wsum.shape[0]
        if len(variances) != n_var:
            raise ValueError(f"expected {n_var} variance planes, got {len(variances)}")
        if self.weight_type == "exptime" and exposure_time is None:
            raise ValueError("exptime weights need the exposure_time")
        if self.weight_type == "ivm" and variances[self.rnoise] is None:
            raise ValueError("ivm weights need the read-noise variance plane")
        n_in = next(np.size(v) for v in variances if v is not None)
        matrix, y0, x0, box_ny, box_nx = _box_rows(W, self.out_shape, n_in)
        present = np.array([v is not None for v in variances])
        sigma = np.zeros((n_var, n_in))
        for q in np.flatnonzero(present):
            sigma[q] = np.sqrt(np.asarray(variances[q], dtype=np.float64).ravel())
        time = -1.0 if exposure_time is None else float(exposure_time)
        _add_stcal_variance(matrix.indptr, matrix.indices, matrix.data, y0, x0, box_ny, box_nx, self.out_shape[1],
                            sigma, present, iscale, self.rnoise, self.weight_type == "ivm", time, self._wsum, self._wt)
        self.n_images += 1

    @property
    def out_var(self):
        """list of ndarray, shape out_shape: each variance plane, `fillval` where no exposure adds to it."""
        scaling = 1.0 / (self.pixel_scale_ratio * self.pixel_scale_ratio)
        planes = []
        for wsum, wt in zip(self._wsum, self._wt):
            out = np.full(wsum.size, self.fillval, dtype=np.float64)
            ok = wt > 0.0
            # as stcal: the weights divided separately, not squared first
            out[ok] = wsum[ok] * scaling / wt[ok] / wt[ok]
            planes.append(out.reshape(self.out_shape))
        return planes

    @property
    def out_err(self):
        """ndarray, shape out_shape: ``sqrt`` of the sum of the finite planes, NaN where none is, as stcal's ERR."""
        planes = np.array(self.out_var)
        finite = np.isfinite(planes)
        err = np.sqrt(np.where(finite, planes, 0.0).sum(axis=0))
        err[~finite.any(axis=0)] = np.nan
        return err


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _add_stcal_variance(indptr, indices, values, y0, x0, box_ny, box_nx, nx_out, sigma, present, iscale, rnoise,
                        ivm, time, wsum, wt):
    """Add one exposure's ``e**2 * w**2`` to wsum and ``w`` to wt, for each plane, where stcal does.

    Parameters
    ----------
    indptr, indices, values : ndarray
        CSR arrays of W, one row per pixel of the box.
    y0, x0, box_ny, box_nx : int
        First output row and column and shape of the box.
    nx_out : int
    sigma : ndarray, shape (n_var, n_in)
        Square root of each variance plane.
    present : ndarray of bool, shape (n_var,)
        Planes the exposure has.
    iscale : float
    rnoise : int
        Read-noise plane, for ivm weights.
    ivm : bool
        Weights from the read-noise plane, else ``time``.
    time : float
    wsum, wt : ndarray of float64, shape (n_var, n_out)
    """
    n_var = sigma.shape[0]
    for by in numba.prange(box_ny):
        e = np.empty(n_var)
        for bx in range(box_nx):
            row = by * box_nx + bx
            if indptr[row] == indptr[row + 1]:
                continue
            sw = 0.0
            for k in range(indptr[row], indptr[row + 1]):
                sw += values[k]
            for q in range(n_var):
                s = 0.0
                for k in range(indptr[row], indptr[row + 1]):
                    s += values[k] * (sigma[q, indices[k]] * iscale)
                mean = s / sw
                e[q] = mean * mean
            if ivm:
                w = 1.0 / e[rnoise] if e[rnoise] > 0.0 and np.isfinite(e[rnoise]) else 0.0
            else:
                w = time
            if not w > 0.0:
                continue
            o = (y0 + by) * nx_out + x0 + bx
            for q in range(n_var):
                if present[q] and e[q] >= 0.0 and np.isfinite(e[q]):
                    wsum[q, o] += e[q] * w * w
                    wt[q, o] += w
