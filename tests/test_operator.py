import numpy as np
import pytest

from spizzle import Coadd, drizzle_operator


def identity_pixmap(shape):
    """Pixel map onto an output grid equal to the input grid."""
    y, x = np.indices(shape, dtype=np.float64)
    return np.dstack([x, y])


def rotated_pixmap(shape, degrees, scale, centre):
    """Pixel map of an input rotated by ``degrees`` and scaled by ``scale``, its centre at ``centre`` (x, y)."""
    t = np.deg2rad(degrees)
    y, x = np.indices(shape, dtype=np.float64)
    xc = x - shape[1] / 2
    yc = y - shape[0] / 2
    xo = scale * (np.cos(t) * xc - np.sin(t) * yc) + centre[0]
    yo = scale * (np.sin(t) * xc + np.cos(t) * yc) + centre[1]
    return np.dstack([xo, yo])


def random_image(shape, low, high, seed):
    """Uniform random image between ``low`` and ``high``."""
    return np.random.default_rng(seed).uniform(low, high, shape)


def test_drizzle_operator_of_identity_pixmap():
    """drizzle_operator of a 5x6 identity pixel map without weights is the 30x30 identity matrix."""
    W = drizzle_operator(identity_pixmap((5, 6)), (5, 6))

    np.testing.assert_array_equal(W.toarray(), np.eye(30))


def test_drizzle_operator_puts_weights_on_the_diagonal():
    """drizzle_operator of a 5x6 identity pixel map with weights has the weights on its diagonal."""
    weights = random_image((5, 6), 0.5, 1.5, seed=1)

    W = drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=weights)

    np.testing.assert_array_equal(W.toarray(), np.diag(weights.ravel()))


def test_drizzle_operator_applies_weight_scale():
    """drizzle_operator of a 5x6 identity pixel map with weight_scale 0.25 has 0.25 on its diagonal."""
    W = drizzle_operator(identity_pixmap((5, 6)), (5, 6), weight_scale=0.25)

    np.testing.assert_array_equal(W.toarray(), 0.25 * np.eye(30))


def test_drizzle_operator_leaves_out_zero_weights():
    """drizzle_operator of a 5x6 identity pixel map with 4 zero weights has 26 stored entries."""
    weights = np.ones((5, 6))
    weights[1, 2:4] = 0.0
    weights[3, 0:2] = 0.0

    W = drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=weights)

    assert W.nnz == 26


def test_drizzle_operator_rejects_weights_of_another_shape():
    """drizzle_operator of a 5x6 pixel map with 5x5 weights raises ValueError."""
    with pytest.raises(ValueError, match="weights shape"):
        drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=np.ones((5, 5)))


def test_drizzle_operator_full_weight_at_the_output_edge():
    """drizzle_operator of an input extending 9.7 pixels past the left output edge gives output column 0 weight 1.

    drizzle gives it 0.7, because it skips input pixels centred outside the output frame.
    """
    pixmap = identity_pixmap((12, 20))
    pixmap[..., 0] -= 9.7

    W = drizzle_operator(pixmap, (12, 10))

    np.testing.assert_allclose((W @ np.ones(240)).reshape(12, 10)[:, 0], np.ones(12), rtol=1e-14)


def test_coadd_of_identity_returns_the_data():
    """Coadd.add of data through an identity operator gives out_img equal to the data."""
    data = random_image((5, 6), 10.0, 11.0, seed=2)
    coadd = Coadd((5, 6))

    coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), data)

    np.testing.assert_array_equal(coadd.out_img, data)


def test_coadd_of_identity_returns_the_weights():
    """Coadd.add through an identity operator with weights gives out_wht equal to the weights."""
    weights = random_image((5, 6), 0.5, 1.5, seed=3)
    coadd = Coadd((5, 6))

    coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=weights), np.ones((5, 6)))

    np.testing.assert_array_equal(coadd.out_wht, weights)


def test_coadd_of_identity_returns_the_variance():
    """Coadd.add with one variance plane through an identity operator gives out_var equal to that plane."""
    var = random_image((5, 6), 1.0, 2.0, seed=4)
    coadd = Coadd((5, 6), n_var=1)

    coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), np.ones((5, 6)), variances=[var])

    np.testing.assert_array_equal(coadd.out_var[0], var)


def test_coadd_of_two_exposures_is_the_weighted_mean():
    """Coadd.add of data 1 with weight 1 and data 4 with weight 2 on the same grid gives out_img 3."""
    pixmap = identity_pixmap((5, 6))
    coadd = Coadd((5, 6))

    coadd.add(drizzle_operator(pixmap, (5, 6)), np.full((5, 6), 1.0))
    coadd.add(drizzle_operator(pixmap, (5, 6), weight_scale=2.0), np.full((5, 6), 4.0))

    np.testing.assert_allclose(coadd.out_img, np.full((5, 6), 3.0), rtol=1e-15)


def test_coadd_variance_of_two_exposures():
    """Coadd.add of variance 1 with weight 1 and variance 4 with weight 2 gives out_var (1 + 4 * 4) / 9."""
    pixmap = identity_pixmap((5, 6))
    coadd = Coadd((5, 6), n_var=1)

    coadd.add(drizzle_operator(pixmap, (5, 6)), np.ones((5, 6)), variances=[np.full((5, 6), 1.0)])
    coadd.add(drizzle_operator(pixmap, (5, 6), weight_scale=2.0), np.ones((5, 6)), variances=[np.full((5, 6), 4.0)])

    np.testing.assert_allclose(coadd.out_var[0], np.full((5, 6), 17.0 / 9.0), rtol=1e-15)


def test_coadd_applies_iscale_to_the_data():
    """Coadd.add with iscale 2.5 through an identity operator gives out_img 2.5 times the data."""
    data = random_image((5, 6), 10.0, 11.0, seed=5)
    coadd = Coadd((5, 6))

    coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), data, iscale=2.5)

    np.testing.assert_array_equal(coadd.out_img, data * 2.5)


def test_coadd_applies_iscale_squared_to_the_variance():
    """Coadd.add with iscale 2.5 through an identity operator gives out_var 6.25 times the variance."""
    var = random_image((5, 6), 1.0, 2.0, seed=6)
    coadd = Coadd((5, 6), n_var=1)

    coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), np.ones((5, 6)), variances=[var], iscale=2.5)

    np.testing.assert_array_equal(coadd.out_var[0], var * 6.25)


def test_coadd_fills_pixels_without_weight():
    """Coadd.add of an input covering output columns 0-5 of 8, with fillval -1, gives -1 in columns 6 and 7."""
    coadd = Coadd((5, 8), fillval=-1.0)

    coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 8)), np.ones((5, 6)))

    np.testing.assert_array_equal(coadd.out_img[:, 6:], np.full((5, 2), -1.0))


def test_coadd_ignores_nan_data_with_zero_weight():
    """Coadd.add of data with a NaN pixel of weight 0 gives an output without NaN and fillval 0 there."""
    data = np.ones((5, 6))
    data[2, 3] = np.nan
    weights = np.ones((5, 6))
    weights[2, 3] = 0.0
    expected = np.ones((5, 6))
    expected[2, 3] = 0.0
    coadd = Coadd((5, 6), fillval=0.0)

    coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=weights), data)

    np.testing.assert_array_equal(coadd.out_img, expected)


def test_coadd_propagates_nan_data_with_weight():
    """Coadd.add of data with a NaN pixel of weight 1 gives NaN in that output pixel, as drizzle does."""
    data = np.ones((5, 6))
    data[2, 3] = np.nan
    coadd = Coadd((5, 6))

    coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), data)

    np.testing.assert_array_equal(np.isnan(coadd.out_img), np.isnan(data))


def test_coadd_context_of_two_exposures():
    """Coadd.add of exposures covering output columns 0-3 and 2-5 gives context 1, 1, 3, 3, 2, 2 along a row."""
    coadd = Coadd((4, 6))
    left = identity_pixmap((4, 4))
    right = identity_pixmap((4, 4)) + [2.0, 0.0]

    coadd.add(drizzle_operator(left, (4, 6)), np.ones((4, 4)))
    coadd.add(drizzle_operator(right, (4, 6)), np.ones((4, 4)))

    np.testing.assert_array_equal(coadd.out_ctx, np.tile([1, 1, 3, 3, 2, 2], (1, 4, 1)))


def test_coadd_context_of_32_exposures_sets_all_bits():
    """Coadd.add of 32 exposures on the same grid gives one int32 context plane with all bits set, -1."""
    W = drizzle_operator(identity_pixmap((3, 4)), (3, 4))
    coadd = Coadd((3, 4))

    for _ in range(32):
        coadd.add(W, np.ones((3, 4)))

    np.testing.assert_array_equal(coadd.out_ctx, np.full((1, 3, 4), -1, dtype=np.int32))


def test_coadd_context_of_33_exposures_adds_a_plane():
    """Coadd.add of 33 exposures on the same grid gives a second context plane with bit 0 set."""
    W = drizzle_operator(identity_pixmap((3, 4)), (3, 4))
    coadd = Coadd((3, 4))

    for _ in range(33):
        coadd.add(W, np.ones((3, 4)))

    np.testing.assert_array_equal(coadd.out_ctx[1], np.ones((3, 4), dtype=np.int32))


def test_coadd_rejects_operator_of_another_shape():
    """Coadd.add of data of 5x6 pixels with an operator built for 4x6 input pixels raises ValueError."""
    coadd = Coadd((5, 6))

    with pytest.raises(ValueError, match="operator shape"):
        coadd.add(drizzle_operator(identity_pixmap((4, 6)), (5, 6)), np.ones((5, 6)))


def test_coadd_rejects_missing_variance_planes():
    """Coadd.add without variances on a Coadd made with n_var=1 raises ValueError."""
    coadd = Coadd((5, 6), n_var=1)

    with pytest.raises(ValueError, match="variance planes"):
        coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 6)), np.ones((5, 6)))


def test_flux_is_conserved():
    """Coadd.add of a rotated, scaled exposure inside the output gives sum(out_img * out_wht) = sum(data * weights)."""
    data = random_image((30, 40), 10.0, 20.0, seed=7)
    weights = random_image((30, 40), 0.5, 1.5, seed=8)
    pixmap = rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0))
    coadd = Coadd((100, 110), fillval=0.0)

    coadd.add(drizzle_operator(pixmap, (100, 110), pixfrac=0.8, weights=weights), data)

    np.testing.assert_allclose(np.sum(coadd.out_img * coadd.out_wht), np.sum(data * weights), rtol=1e-10)
