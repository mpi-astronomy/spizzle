import numba
import numpy as np
import pytest

from spizzle.geometry import _boxer, _nintd, corners, jacobian, overlap_triplets


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


def expected_corners(shape, h):
    """Corners of the pixels of an identity pixel map shrunk to half-width h, in drizzle's order."""
    y, x = np.indices(shape, dtype=np.float64)
    return np.stack([
        np.dstack([x - h, y + h]),
        np.dstack([x + h, y + h]),
        np.dstack([x + h, y - h]),
        np.dstack([x - h, y - h]),
    ], axis=2)


@pytest.mark.parametrize("x, expected", [
    (0.5, 1.0),
    (-0.5, -1.0),
    (1.5, 2.0),
    (2.5, 3.0),
    (-2.5, -3.0),
    (0.49999999999999994, 0.0),
    (3.7, 4.0),
    (-1.2, -1.0),
])
def test_nintd_rounds_halves_away_from_zero(x, expected):
    """_nintd(x) gives the nearest integer, rounding halves away from zero as C round() does."""
    assert _nintd(x) == expected


def test_boxer_unit_square_on_the_pixel():
    """_boxer for output pixel (3, 2) and the clockwise unit square around its centre gives 1."""
    assert _boxer(3.0, 2.0, 2.5, 2.5, 3.5, 2.5, 3.5, 1.5, 2.5, 1.5) == 1.0


def test_boxer_unit_square_shifted_by_a_quarter():
    """_boxer for output pixel (3, 2) and a unit square shifted +0.25 in x gives 0.75."""
    assert _boxer(3.0, 2.0, 2.75, 2.5, 3.75, 2.5, 3.75, 1.5, 2.75, 1.5) == 0.75


def test_boxer_anticlockwise_square_is_negative():
    """_boxer for output pixel (3, 2) and its unit square in anticlockwise order gives -1."""
    assert _boxer(3.0, 2.0, 2.5, 1.5, 3.5, 1.5, 3.5, 2.5, 2.5, 2.5) == -1.0


def test_boxer_diamond_inside_the_pixel():
    """_boxer for output pixel (3, 2) and the diamond with vertices on the pixel's edge midpoints gives 0.5."""
    assert _boxer(3.0, 2.0, 3.0, 2.5, 3.5, 2.0, 3.0, 1.5, 2.5, 2.0) == 0.5


def test_boxer_diamond_on_the_right_edge():
    """_boxer for output pixel (3, 2) and the same diamond centred on the pixel's right edge gives 0.25."""
    assert _boxer(3.0, 2.0, 3.5, 2.5, 4.0, 2.0, 3.5, 1.5, 3.0, 2.0) == 0.25


def test_boxer_square_outside_the_pixel():
    """_boxer for output pixel (3, 2) and a unit square around (5, 2) gives 0."""
    assert _boxer(3.0, 2.0, 4.5, 2.5, 5.5, 2.5, 5.5, 1.5, 4.5, 1.5) == 0.0


@pytest.mark.parametrize("pixfrac", [1.0, 0.5])
def test_corners_of_identity_pixmap(pixfrac):
    """corners of an 8x9 identity pixel map are at x -/+ pixfrac/2, y -/+ pixfrac/2 of each pixel, clockwise from top left."""
    c = corners(identity_pixmap((8, 9)), pixfrac)

    np.testing.assert_array_equal(c, expected_corners((8, 9), pixfrac / 2))


def test_corners_nan_in_pixmap():
    """corners of an 8x8 identity pixel map with NaN at pixel (3, 3) are NaN for the 3x3 pixels around it only."""
    pixmap = identity_pixmap((8, 8))
    pixmap[3, 3] = np.nan
    expected = np.zeros((8, 8), dtype=bool)
    expected[2:5, 2:5] = True

    c = corners(pixmap, 1.0)

    np.testing.assert_array_equal(np.isnan(c).any(axis=(2, 3)), expected)


def test_corners_rejects_pixmap_without_two_planes():
    """corners of a pixel map of shape (4, 4, 3) raises ValueError."""
    with pytest.raises(ValueError, match="shape"):
        corners(np.zeros((4, 4, 3)), 1.0)


def test_corners_rejects_pixmap_of_one_row():
    """corners of a pixel map of shape (1, 4, 2) raises ValueError."""
    with pytest.raises(ValueError, match="at least 2x2"):
        corners(np.zeros((1, 4, 2)), 1.0)


def test_corners_rejects_tiny_pixfrac():
    """corners with pixfrac 0.001, where drizzle switches to the point kernel, raises ValueError."""
    with pytest.raises(ValueError, match="pixfrac"):
        corners(identity_pixmap((4, 4)), 0.001)


def test_jacobian_of_identity_pixmap():
    """jacobian of identity pixel map corners with pixfrac 0.8 is 0.64 everywhere."""
    jaco = jacobian(corners(identity_pixmap((6, 7)), 0.8))

    np.testing.assert_allclose(jaco, np.full((6, 7), 0.64), rtol=1e-14)


def test_jacobian_of_flipped_pixmap_is_negative():
    """jacobian of a pixel map that flips x, with pixfrac 1, is -1 everywhere."""
    pixmap = identity_pixmap((6, 7))
    pixmap[..., 0] = 10.0 - pixmap[..., 0]

    jaco = jacobian(corners(pixmap, 1.0))

    np.testing.assert_array_equal(jaco, np.full((6, 7), -1.0))


def test_overlap_triplets_of_identity_pixmap():
    """overlap_triplets of an identity pixel map with pixfrac 1 gives each input pixel its own output pixel with area 1."""
    in_idx, out_idx, area = overlap_triplets(corners(identity_pixmap((6, 7)), 1.0), (6, 7))

    np.testing.assert_array_equal(in_idx, np.arange(42))
    np.testing.assert_array_equal(out_idx, np.arange(42))
    np.testing.assert_array_equal(area, np.ones(42))


def test_overlap_triplets_half_pixel_shift():
    """overlap_triplets of a 2x3 input shifted +0.5 pixel in x onto a 2x4 output splits each input pixel into two halves."""
    pixmap = identity_pixmap((2, 3))
    pixmap[..., 0] += 0.5

    in_idx, out_idx, area = overlap_triplets(corners(pixmap, 1.0), (2, 4))

    np.testing.assert_array_equal(in_idx, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    np.testing.assert_array_equal(out_idx, [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7])
    np.testing.assert_array_equal(area, np.full(12, 0.5))


def test_overlap_triplets_are_in_input_pixel_order():
    """overlap_triplets of a rotated, scaled pixel map gives input indices in non-decreasing order."""
    c = corners(rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0)), 0.8)

    in_idx, _, _ = overlap_triplets(c, (100, 110))

    assert np.all(np.diff(in_idx) >= 0)


def test_overlap_triplets_areas_sum_to_jacobian():
    """overlap_triplets of a rotated, scaled pixel map inside the output gives areas that sum to each pixel's jacobian."""
    c = corners(rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0)), 0.8)

    in_idx, _, area = overlap_triplets(c, (100, 110))

    np.testing.assert_allclose(np.bincount(in_idx, area, minlength=1200), jacobian(c).ravel(), rtol=1e-13)


def test_overlap_triplets_input_outside_output():
    """overlap_triplets of an input mapped entirely beyond the output gives no triplets."""
    pixmap = identity_pixmap((6, 7)) + 20.0

    in_idx, _, _ = overlap_triplets(corners(pixmap, 1.0), (6, 7))

    assert in_idx.size == 0


def test_overlap_triplets_skip_nan_corners():
    """overlap_triplets of an 8x8 identity pixel map with NaN at pixel (3, 3) gives no triplets for the 3x3 pixels around it."""
    pixmap = identity_pixmap((8, 8))
    pixmap[3, 3] = np.nan
    skipped = np.ravel_multi_index(np.indices((3, 3)).reshape(2, -1) + 2, (8, 8))

    in_idx, _, _ = overlap_triplets(corners(pixmap, 1.0), (8, 8))

    np.testing.assert_array_equal(in_idx, np.setdiff1d(np.arange(64), skipped))


def test_overlap_triplets_cover_output_edge():
    """overlap_triplets of an input whose column 9 is centred at x = -0.7 gives that column area 0.3 in output column 0."""
    pixmap = identity_pixmap((3, 20))
    pixmap[..., 0] -= 9.7
    in_idx, out_idx, area = overlap_triplets(corners(pixmap, 1.0), (3, 10))

    column_9_to_0 = (in_idx % 20 == 9) & (out_idx % 10 == 0)

    np.testing.assert_allclose(area[column_9_to_0], [0.3, 0.3, 0.3], rtol=1e-12)


def test_overlap_triplets_same_on_one_thread():
    """overlap_triplets of a rotated pixel map on one numba thread gives the same triplets as on all threads."""
    c = corners(rotated_pixmap((30, 40), 33.0, 1.6, (55.0, 50.0)), 0.8)
    in_all, out_all, area_all = overlap_triplets(c, (100, 110))

    numba.set_num_threads(1)
    try:
        in_one, out_one, area_one = overlap_triplets(c, (100, 110))
    finally:
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    np.testing.assert_array_equal(in_one, in_all)
    np.testing.assert_array_equal(out_one, out_all)
    np.testing.assert_array_equal(area_one, area_all)
