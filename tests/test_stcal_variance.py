"""StcalVariance against stcal's variance arithmetic, with drizzle's tdriz for the resampled errors.

The reference below follows stcal 1.20's ``resample_variance_arrays`` and
``finalize_resample_variance`` line for line.  drizzle must accept float64
output arrays, from jdavies-st/drizzle@float64-phase2-c-accepts-float64
until that is released.
"""
import numpy as np
import pytest
from drizzle import cdrizzle

from spizzle import StcalVariance, drizzle_operator

SHAPE, OUT_SHAPE = (12, 15), (40, 45)
PIXFRAC = 0.8


def identity_pixmap(shape):
    """Pixel map onto an output grid equal to the input grid."""
    y, x = np.indices(shape, dtype=np.float64)
    return np.dstack([x, y])


def rotated_pixmap(shape, degrees, scale, center):
    """Pixel map of an input rotated by ``degrees`` and scaled by ``scale``, its center at ``center`` (x, y)."""
    t = np.deg2rad(degrees)
    y, x = np.indices(shape, dtype=np.float64)
    xc = x - shape[1] / 2
    yc = y - shape[0] / 2
    xo = scale * (np.cos(t) * xc - np.sin(t) * yc) + center[0]
    yo = scale * (np.sin(t) * xc + np.cos(t) * yc) + center[1]
    return np.dstack([xo, yo])


def random_image(shape, low, high, seed):
    """Uniform random image between ``low`` and ``high``."""
    return np.random.default_rng(seed).uniform(low, high, shape)


def two_exposures():
    """Pixel maps, weights, three variance planes and exposure times of two rotated 12x15 exposures on a 40x45 output."""
    pixmaps = [rotated_pixmap(SHAPE, 33.0, 1.6, (22.0, 20.0)), rotated_pixmap(SHAPE, 71.0, 1.6, (24.0, 21.0))]
    weights = [random_image(SHAPE, 0.5, 1.5, seed=3), random_image(SHAPE, 0.5, 1.5, seed=4)]
    variances = [[random_image(SHAPE, 1.0, 2.0, seed=10 * k + q) for q in range(3)] for k in range(2)]
    return list(zip(pixmaps, weights, variances, [100.0, 300.0]))


def resampled_error_squared(var, weights, pixmap):
    """drizzle's weighted mean of sqrt(var) on the output, squared, NaN where nothing lands: stcal's resampled variance."""
    out = np.full(OUT_SHAPE, np.nan)
    cdrizzle.tdriz(np.sqrt(var), weights, pixmap, out, np.zeros(OUT_SHAPE), np.zeros(OUT_SHAPE, dtype=np.int32),
                   uniqid=1, iscale=1.0, pixfrac=PIXFRAC, kernel="square", in_units="cps", expscale=1.0, wtscale=1.0,
                   fillstr=str(np.nan))
    return out ** 2


def stcal_reference(exposures, weight_type, pixel_scale_ratio=1.0):
    """stcal's variance planes for imaging, read noise first."""
    wsum = np.full((3,) + OUT_SHAPE, np.nan)
    wt = np.zeros((3,) + OUT_SHAPE)
    for pixmap, weights, variances, time in exposures:
        resampled = [resampled_error_squared(var, weights, pixmap) for var in variances]
        if weight_type == "ivm":
            rn = resampled[0]
            mask = (rn > 0) & np.isfinite(rn)
            weight = np.zeros(OUT_SHAPE)
            weight[mask] = np.reciprocal(rn[mask])
        else:
            weight = np.full(OUT_SHAPE, time)
        for q in range(3):
            var = resampled[q]
            mask = (var >= 0) & np.isfinite(var) & (weight > 0)
            wsum[q][mask] = np.nansum([wsum[q][mask], var[mask] * weight[mask] * weight[mask]], axis=0)
            wt[q][mask] += weight[mask]
    with np.errstate(invalid="ignore", divide="ignore"):
        return wsum * (1.0 / pixel_scale_ratio ** 2) / wt / wt


def stcal_variance_of(exposures, weight_type, pixel_scale_ratio=1.0):
    """StcalVariance of the exposures, with their operators at PIXFRAC."""
    sv = StcalVariance(OUT_SHAPE, weight_type=weight_type, pixel_scale_ratio=pixel_scale_ratio)
    for pixmap, weights, variances, time in exposures:
        sv.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=PIXFRAC, weights=weights), variances, exposure_time=time)
    return sv


def test_stcal_variance_with_ivm_weights_matches_stcal():
    """StcalVariance.out_var of two rotated, weighted exposures with ivm weights equals stcal's planes, to 1e-12."""
    exposures = two_exposures()

    sv = stcal_variance_of(exposures, "ivm")

    np.testing.assert_allclose(sv.out_var, stcal_reference(exposures, "ivm"), rtol=1e-12, equal_nan=True)


def test_stcal_variance_with_exptime_weights_matches_stcal():
    """StcalVariance.out_var of two exposures of 100 s and 300 s with exptime weights equals stcal's planes, to 1e-12."""
    exposures = two_exposures()

    sv = stcal_variance_of(exposures, "exptime")

    np.testing.assert_allclose(sv.out_var, stcal_reference(exposures, "exptime"), rtol=1e-12, equal_nan=True)


def test_stcal_variance_divides_by_the_pixel_scale_ratio_squared():
    """StcalVariance.out_var with pixel_scale_ratio 0.5 equals stcal's planes for that ratio, 4 times larger, to 1e-12."""
    exposures = two_exposures()

    sv = stcal_variance_of(exposures, "ivm", pixel_scale_ratio=0.5)

    np.testing.assert_allclose(sv.out_var, stcal_reference(exposures, "ivm", 0.5), rtol=1e-12, equal_nan=True)


def test_stcal_err_is_the_root_of_the_summed_planes():
    """StcalVariance.out_err of two rotated exposures is sqrt of the sum of stcal's planes, NaN where none is finite."""
    exposures = two_exposures()
    planes = stcal_reference(exposures, "ivm")
    expected = np.sqrt(np.nansum(planes, axis=0))
    expected[np.all(np.isnan(planes), axis=0)] = np.nan

    sv = stcal_variance_of(exposures, "ivm")

    np.testing.assert_allclose(sv.out_err, expected, rtol=1e-12, equal_nan=True)


def test_stcal_variance_of_identity_is_the_input_variance():
    """StcalVariance.out_var of one exposure on an identical grid gives each input variance plane back."""
    variances = [random_image((5, 6), 1.0, 2.0, seed=q) for q in range(3)]
    sv = StcalVariance((5, 6))

    sv.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), variances)

    np.testing.assert_allclose(sv.out_var, variances, rtol=1e-12)


def test_stcal_variance_leaves_out_nan_variance():
    """StcalVariance.out_var of one identity exposure whose flat variance is NaN at (2, 3) is NaN there in that plane only."""
    variances = [np.ones((5, 6)), np.ones((5, 6)), np.ones((5, 6))]
    variances[1][2, 3] = np.nan
    sv = StcalVariance((5, 6))

    sv.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), variances)

    assert [np.isnan(plane[2, 3]) for plane in sv.out_var] == [False, True, False]


def test_stcal_variance_needs_the_exposure_time_for_exptime_weights():
    """StcalVariance.add with exptime weights and no exposure_time raises ValueError."""
    sv = StcalVariance((5, 6), weight_type="exptime")

    with pytest.raises(ValueError, match="exposure_time"):
        sv.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), [np.ones((5, 6))] * 3)


def test_stcal_variance_rejects_another_weight_type():
    """StcalVariance with weight_type 'ivm-sky' raises ValueError."""
    with pytest.raises(ValueError, match="weight_type"):
        StcalVariance((5, 6), weight_type="ivm-sky")
