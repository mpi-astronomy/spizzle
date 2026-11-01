import numpy as np
import pytest

from spizzle import Coadd, Covariance, drizzle_operator


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


def dense_covariance_planes(ops, variances, out_shape, offsets):
    """Covariance planes from the dense covariance matrix sum(W var W.T) / outer(wsum, wsum), NaN where undefined."""
    ny, nx = out_shape
    n_out = ny * nx
    num = np.zeros((n_out, n_out))
    wsum = np.zeros(n_out)
    for op, var in zip(ops, variances):
        W = op.full().toarray()
        num += (W * var.ravel()) @ W.T
        wsum += W.sum(axis=1)
    planes = np.full((len(offsets), ny, nx), np.nan)
    for k, (dy, dx) in enumerate(offsets):
        for y in range(ny - dy):
            for x in range(max(0, -dx), nx - max(0, dx)):
                o = y * nx + x
                o2 = (y + dy) * nx + x + dx
                if wsum[o] != 0 and wsum[o2] != 0:
                    planes[k, y, x] = num[o, o2] / (wsum[o] * wsum[o2])
    return planes


def test_covariance_offsets_for_radius_1():
    """Covariance with radius 1 keeps the offsets (0, 0), (0, 1), (1, -1), (1, 0) and (1, 1)."""
    cov = Covariance((5, 6), radius=1)

    np.testing.assert_array_equal(cov.offsets, [[0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])


def test_covariance_variance_of_half_pixel_shift():
    """Covariance of an input shifted +0.5 pixel in x, variance 1, gives variance 0.5 in output columns 1-7."""
    pixmap = identity_pixmap((6, 8)) + [0.5, 0.0]
    cov = Covariance((6, 9), radius=1)

    cov.add(drizzle_operator(pixmap, (6, 9)), np.ones((6, 8)))

    np.testing.assert_allclose(cov.out_var[:, 1:8], np.full((6, 7), 0.5), rtol=1e-15)


def test_covariance_of_half_pixel_shift_with_the_next_column():
    """Covariance of an input shifted +0.5 pixel in x, variance 1, gives covariance 0.25 of output columns x and x + 1, x 1-6."""
    pixmap = identity_pixmap((6, 8)) + [0.5, 0.0]
    cov = Covariance((6, 9), radius=1)

    cov.add(drizzle_operator(pixmap, (6, 9)), np.ones((6, 8)))

    np.testing.assert_allclose(cov.out_cov[1][:, 1:7], np.full((6, 6), 0.25), rtol=1e-15)


def test_covariance_of_half_pixel_shift_with_the_next_row():
    """Covariance of an input shifted +0.5 pixel in x gives covariance 0 between output rows y and y + 1."""
    pixmap = identity_pixmap((6, 8)) + [0.5, 0.0]
    cov = Covariance((6, 9), radius=1)

    cov.add(drizzle_operator(pixmap, (6, 9)), np.ones((6, 8)))

    np.testing.assert_array_equal(cov.out_cov[3][:5], np.zeros((5, 9)))


def test_covariance_variance_matches_coadd():
    """Covariance.out_var of a rotated, weighted exposure equals Coadd.out_var of the same variance bit for bit."""
    pixmap = rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0))
    op = drizzle_operator(pixmap, (100, 110), pixfrac=0.8, weights=random_image((30, 40), 0.5, 1.5, seed=1))
    var = random_image((30, 40), 1.0, 2.0, seed=2)
    coadd = Coadd((100, 110), n_var=1)
    cov = Covariance((100, 110), radius=2)

    coadd.add(op, np.ones((30, 40)), variances=[var])
    cov.add(op, var)

    np.testing.assert_array_equal(cov.out_var, coadd.out_var[0])


def test_covariance_of_two_exposures_matches_dense_covariance():
    """Covariance with radius 3 of two rotated, weighted exposures equals dense W var W.T over the weights, to 1e-12."""
    shape, out_shape = (12, 15), (40, 45)
    pixmaps = [rotated_pixmap(shape, 33.0, 1.6, (22.0, 20.0)), rotated_pixmap(shape, 71.0, 1.6, (24.0, 21.0))]
    weights = [random_image(shape, 0.5, 1.5, seed=3), random_image(shape, 0.5, 1.5, seed=4)]
    variances = [random_image(shape, 1.0, 2.0, seed=5), random_image(shape, 1.0, 2.0, seed=6)]
    ops = [drizzle_operator(p, out_shape, pixfrac=0.8, weights=w) for p, w in zip(pixmaps, weights)]
    cov = Covariance(out_shape, radius=3)
    expected = dense_covariance_planes(ops, variances, out_shape, cov.offsets)

    for op, var in zip(ops, variances):
        cov.add(op, var)

    np.testing.assert_allclose(cov.out_cov, expected, rtol=1e-12, atol=0, equal_nan=True)


def test_covariance_is_zero_beyond_the_footprint():
    """Covariance with radius 4 of an input rotated 33 deg with 1.28-pixel drops is zero at offsets of 3 or more pixels."""
    pixmap = rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0))
    cov = Covariance((100, 110), radius=4)
    far = np.abs(cov.offsets).max(axis=1) >= 3

    cov.add(drizzle_operator(pixmap, (100, 110), pixfrac=0.8), np.ones((30, 40)))

    far_planes = cov.out_cov[far]
    np.testing.assert_array_equal(far_planes[np.isfinite(far_planes)], 0.0)


def test_covariance_applies_iscale_squared():
    """Covariance.add with iscale 2 gives 4 times the covariance with iscale 1."""
    pixmap = rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0))
    op = drizzle_operator(pixmap, (100, 110), pixfrac=0.8)
    var = random_image((30, 40), 1.0, 2.0, seed=7)
    scaled = Covariance((100, 110), radius=2)
    unscaled = Covariance((100, 110), radius=2)

    scaled.add(op, var, iscale=2.0)
    unscaled.add(op, var)

    np.testing.assert_allclose(scaled.out_cov, 4.0 * unscaled.out_cov, rtol=1e-15, equal_nan=True)


def test_covariance_accepts_operator_on_the_whole_grid():
    """Covariance.add of Operator.full() gives the same covariance as Covariance.add of the Operator."""
    pixmap = rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0))
    op = drizzle_operator(pixmap, (100, 110), pixfrac=0.8)
    var = random_image((30, 40), 1.0, 2.0, seed=8)
    on_box = Covariance((100, 110), radius=2)
    on_grid = Covariance((100, 110), radius=2)

    on_box.add(op, var)
    on_grid.add(op.full(), var)

    np.testing.assert_array_equal(on_grid.out_cov, on_box.out_cov)


def test_covariance_fills_pixels_without_weight():
    """Covariance of an input covering output columns 0-5 of 8, with fillval -1, gives -1 for the variance in columns 6 and 7."""
    cov = Covariance((5, 8), radius=1, fillval=-1.0)

    cov.add(drizzle_operator(identity_pixmap((5, 6)), (5, 8)), np.ones((5, 6)))

    np.testing.assert_array_equal(cov.out_var[:, 6:], np.full((5, 2), -1.0))


def test_covariance_rejects_operator_for_another_output_shape():
    """Covariance.add on a 5x6 Covariance of an operator built for a 5x7 output raises ValueError."""
    cov = Covariance((5, 6))

    with pytest.raises(ValueError, match="output shape"):
        cov.add(drizzle_operator(identity_pixmap((5, 6)), (5, 7)), np.ones((5, 6)))


def test_covariance_rejects_negative_radius():
    """Covariance with radius -1 raises ValueError."""
    with pytest.raises(ValueError, match="radius"):
        Covariance((5, 6), radius=-1)
