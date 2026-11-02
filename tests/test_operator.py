import numba
import numpy as np
import pytest
from scipy import sparse

from spizzle import Coadd, build_operator, corners, drizzle_operator, jacobian, overlap_triplets


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


def scipy_operator(triplets, jaco, out_shape, weights):
    """The operator as scipy builds it from the triplets: area * weight / jaco, zero entries left out."""
    in_idx, out_idx, area = triplets
    values = area * (weights / jaco).ravel()[in_idx]
    keep = values != 0.0
    return sparse.csr_array((values[keep], (out_idx[keep], in_idx[keep])), shape=(out_shape[0] * out_shape[1], jaco.size))


def test_drizzle_operator_of_identity_pixmap():
    """drizzle_operator of a 5x6 identity pixel map without weights is, on the whole grid, the 30x30 identity matrix."""
    W = drizzle_operator(identity_pixmap((5, 6)), (5, 6)).full()

    np.testing.assert_array_equal(W.toarray(), np.eye(30))


def test_drizzle_operator_puts_weights_on_the_diagonal():
    """drizzle_operator of a 5x6 identity pixel map with weights has the weights on its diagonal."""
    weights = random_image((5, 6), 0.5, 1.5, seed=1)

    W = drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=weights).full()

    np.testing.assert_array_equal(W.toarray(), np.diag(weights.ravel()))


def test_drizzle_operator_applies_weight_scale():
    """drizzle_operator of a 5x6 identity pixel map with weight_scale 0.25 has 0.25 on its diagonal."""
    W = drizzle_operator(identity_pixmap((5, 6)), (5, 6), weight_scale=0.25).full()

    np.testing.assert_array_equal(W.toarray(), 0.25 * np.eye(30))


def test_drizzle_operator_leaves_out_zero_weights():
    """drizzle_operator of a 5x6 identity pixel map with 4 zero weights has 26 stored entries."""
    weights = np.ones((5, 6))
    weights[1, 2:4] = 0.0
    weights[3, 0:2] = 0.0

    op = drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=weights)

    assert op.matrix.nnz == 26


def test_drizzle_operator_box_of_shifted_input():
    """drizzle_operator of a 5x6 identity pixel map shifted by (+3, +2) onto a 12x14 output has box rows 2-6, columns 3-8."""
    op = drizzle_operator(identity_pixmap((5, 6)) + [3.0, 2.0], (12, 14))

    assert op.box == (slice(2, 7), slice(3, 9))


def test_drizzle_operator_box_shape_of_shifted_input():
    """drizzle_operator of a 5x6 identity pixel map shifted by (+3, +2) onto a 12x14 output has box_shape (5, 6)."""
    op = drizzle_operator(identity_pixmap((5, 6)) + [3.0, 2.0], (12, 14))

    assert op.box_shape == (5, 6)


def test_drizzle_operator_matrix_of_shifted_input():
    """drizzle_operator of a 5x6 identity pixel map shifted by (+3, +2) onto a 12x14 output has the identity on its box."""
    op = drizzle_operator(identity_pixmap((5, 6)) + [3.0, 2.0], (12, 14))

    np.testing.assert_array_equal(op.matrix.toarray(), np.eye(30))


def test_operator_full_of_shifted_input():
    """Operator.full of a 5x6 identity pixel map shifted by (+3, +2) onto 12x14 maps input (y, x) to output (y + 2, x + 3)."""
    y, x = np.indices((5, 6))
    expected = np.zeros((12 * 14, 30))
    expected[((y + 2) * 14 + x + 3).ravel(), np.arange(30)] = 1.0

    op = drizzle_operator(identity_pixmap((5, 6)) + [3.0, 2.0], (12, 14))

    np.testing.assert_array_equal(op.full().toarray(), expected)


def test_drizzle_operator_box_leaves_out_columns_of_zero_weight():
    """drizzle_operator of a 5x6 identity pixel map with zero weights in input column 5 has box columns 0-4."""
    weights = np.ones((5, 6))
    weights[:, 5] = 0.0

    op = drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=weights)

    assert op.box == (slice(0, 5), slice(0, 5))


def test_drizzle_operator_of_input_outside_output_has_no_rows():
    """drizzle_operator of an input mapped entirely beyond the output gives a matrix of shape (0, 42)."""
    op = drizzle_operator(identity_pixmap((6, 7)) + 20.0, (6, 7))

    assert op.matrix.shape == (0, 42)


def test_drizzle_operator_rejects_weights_of_another_shape():
    """drizzle_operator of a 5x6 pixel map with 5x5 weights raises ValueError."""
    with pytest.raises(ValueError, match="weights shape"):
        drizzle_operator(identity_pixmap((5, 6)), (5, 6), weights=np.ones((5, 5)))


def test_drizzle_operator_region_equals_zero_weights_outside_it():
    """drizzle_operator of a rotated exposure with region rows 5-19, columns 8-30 equals the one with zero weight elsewhere."""
    pixmap = rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0))
    weights = np.zeros((30, 40))
    weights[5:20, 8:31] = 1.0
    expected = drizzle_operator(pixmap, (100, 110), pixfrac=0.8, weights=weights).full()

    W = drizzle_operator(pixmap, (100, 110), pixfrac=0.8, region=(slice(5, 20), slice(8, 31))).full()

    np.testing.assert_array_equal(W.indptr, expected.indptr)
    np.testing.assert_array_equal(W.indices, expected.indices)
    np.testing.assert_array_equal(W.data, expected.data)


def test_drizzle_operator_region_beyond_the_input_is_clipped():
    """drizzle_operator of a rotated 30x40 exposure with region rows and columns 0-99 equals the operator without a region."""
    pixmap = rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0))
    expected = drizzle_operator(pixmap, (100, 110)).full()

    W = drizzle_operator(pixmap, (100, 110), region=(slice(0, 100), slice(0, 100))).full()

    np.testing.assert_array_equal(W.indices, expected.indices)
    np.testing.assert_array_equal(W.data, expected.data)


def test_drizzle_operator_empty_region_has_no_entries():
    """drizzle_operator of an exposure with the empty region rows 3-2 gives an operator without entries."""
    op = drizzle_operator(identity_pixmap((5, 6)), (5, 6), region=(slice(3, 2), slice(None)))

    assert op.matrix.nnz == 0


def test_drizzle_operator_rejects_a_region_with_steps():
    """drizzle_operator with region rows 0:5:2 raises ValueError."""
    with pytest.raises(ValueError, match="step"):
        drizzle_operator(identity_pixmap((5, 6)), (5, 6), region=(slice(0, 5, 2), slice(None)))


def test_build_operator_matches_scipy():
    """build_operator of a rotated, weighted exposure gives the same CSR arrays as scipy builds from its triplets."""
    c = corners(rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0)), 0.8)
    triplets = overlap_triplets(c, (100, 110))
    weights = random_image((30, 40), 0.5, 1.5, seed=9)
    expected = scipy_operator(triplets, jacobian(c), (100, 110), weights)

    W = build_operator(triplets, jacobian(c), (100, 110), weights=weights).full()

    np.testing.assert_array_equal(W.indptr, expected.indptr)
    np.testing.assert_array_equal(W.indices, expected.indices)
    np.testing.assert_array_equal(W.data, expected.data)


def test_build_operator_with_zero_weights_matches_scipy():
    """build_operator of a rotated exposure with 120 zero weights gives the same CSR arrays as scipy without those entries."""
    c = corners(rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0)), 0.8)
    triplets = overlap_triplets(c, (100, 110))
    weights = random_image((30, 40), 0.5, 1.5, seed=10)
    weights.flat[np.random.default_rng(11).choice(weights.size, 120, replace=False)] = 0.0
    expected = scipy_operator(triplets, jacobian(c), (100, 110), weights)

    W = build_operator(triplets, jacobian(c), (100, 110), weights=weights).full()

    np.testing.assert_array_equal(W.indptr, expected.indptr)
    np.testing.assert_array_equal(W.indices, expected.indices)
    np.testing.assert_array_equal(W.data, expected.data)


def test_build_operator_same_on_one_thread():
    """build_operator of a rotated exposure on one numba thread gives the same CSR arrays as on all threads."""
    c = corners(rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0)), 0.8)
    triplets = overlap_triplets(c, (100, 110))
    W_all = build_operator(triplets, jacobian(c), (100, 110)).full()

    numba.set_num_threads(1)
    try:
        W_one = build_operator(triplets, jacobian(c), (100, 110)).full()
    finally:
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    np.testing.assert_array_equal(W_one.indptr, W_all.indptr)
    np.testing.assert_array_equal(W_one.indices, W_all.indices)
    np.testing.assert_array_equal(W_one.data, W_all.data)


def test_drizzle_operator_full_weight_at_the_output_edge():
    """drizzle_operator of an input extending 9.7 pixels past the left output edge gives output column 0 weight 1.

    drizzle gives it 0.7, because it skips input pixels centered outside the output frame.
    """
    pixmap = identity_pixmap((12, 20))
    pixmap[..., 0] -= 9.7

    W = drizzle_operator(pixmap, (12, 10)).full()

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


def test_coadd_of_operator_equals_coadd_of_its_full_array():
    """Coadd.add of a rotated exposure's Operator gives the same out_img as Coadd.add of Operator.full()."""
    data = random_image((30, 40), 10.0, 20.0, seed=13)
    op = drizzle_operator(rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0)), (100, 110), pixfrac=0.8)
    on_box = Coadd((100, 110))
    on_grid = Coadd((100, 110))

    on_box.add(op, data)
    on_grid.add(op.full(), data)

    np.testing.assert_array_equal(on_box.out_img, on_grid.out_img)


def test_coadd_accepts_csc_operator():
    """Coadd.add of a rotated exposure's operator as a CSC array on the whole grid gives the same out_img as the Operator."""
    data = random_image((30, 40), 10.0, 20.0, seed=12)
    op = drizzle_operator(rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0)), (100, 110), pixfrac=0.8)
    csr = Coadd((100, 110))
    csc = Coadd((100, 110))

    csr.add(op, data)
    csc.add(sparse.csc_array(op.full()), data)

    np.testing.assert_array_equal(csc.out_img, csr.out_img)


def test_coadd_of_input_outside_output_adds_no_weight():
    """Coadd.add of an input mapped entirely beyond the output leaves out_wht zero."""
    coadd = Coadd((6, 7))

    coadd.add(drizzle_operator(identity_pixmap((6, 7)) + 20.0, (6, 7)), np.ones((6, 7)))

    np.testing.assert_array_equal(coadd.out_wht, np.zeros((6, 7)))


def test_coadd_rejects_operator_for_another_output_shape():
    """Coadd.add on a 5x6 Coadd of an operator built for a 5x7 output raises ValueError."""
    coadd = Coadd((5, 6))

    with pytest.raises(ValueError, match="output shape"):
        coadd.add(drizzle_operator(identity_pixmap((5, 6)), (5, 7)), np.ones((5, 6)))


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
