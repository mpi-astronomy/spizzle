"""Comparison with drizzle's bilinear blot.

drizzle must accept float64 arrays in ``tblot``, from
jdavies-st/drizzle@float64-phase2-c-accepts-float64 until that is
released.  The tolerances allow for drizzle builds that contract
multiplications and additions into fused multiply-adds.
"""
import numba
import numpy as np
import pytest
from drizzle import cdrizzle

from spizzle import blot

DATA_SHAPE = (50, 60)
SHAPE = (40, 45)


def random_pixmap(seed=2):
    """Positions over [-2, 62) x [-2, 52): outside, inside and in the last row and column of a 50x60 image."""
    rng = np.random.default_rng(seed)
    pixmap = np.dstack([rng.uniform(-2.0, 62.0, SHAPE), rng.uniform(-2.0, 52.0, SHAPE)])
    pixmap[0, :5, 0] = 59.0 + rng.uniform(0.0, 1.0, 5)
    pixmap[1, :5, 1] = 49.0 + rng.uniform(0.0, 1.0, 5)
    pixmap[2, 0] = (59.5, 49.5)
    return pixmap


def tblot(data, pixmap, out, iscale=1.0, fillval=np.nan):
    """drizzle's bilinear blot of ``data`` into ``out``."""
    cdrizzle.tblot(data, pixmap, out, iscale=iscale, interp="linear", fillval=fillval)
    return out


def test_blot_matches_drizzle():
    """blot of a random 50x60 image at random positions, with iscale 1.7, gives drizzle's tblot result to 1e-12."""
    data = np.random.default_rng(1).normal(size=DATA_SHAPE)
    pixmap = random_pixmap()
    expected = tblot(data, pixmap, np.zeros(SHAPE), iscale=1.7, fillval=-5.0)

    result = blot(data, pixmap, fillval=-5.0, iscale=1.7)

    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_blot_float32_matches_drizzle():
    """blot of a float32 image into a float32 array gives drizzle's tblot result to 1 float32 rounding step."""
    data = np.random.default_rng(1).normal(size=DATA_SHAPE).astype(np.float32)
    pixmap = random_pixmap()
    expected = tblot(data, pixmap, np.zeros(SHAPE, dtype=np.float32))

    result = blot(data, pixmap, out=np.zeros(SHAPE, dtype=np.float32))

    np.testing.assert_array_max_ulp(result, expected, maxulp=1)


def test_blot_interpolates_bilinearly():
    """blot of the image 10 * x + y at (1.25, 2.5) gives 15.0."""
    y, x = np.indices(DATA_SHAPE, dtype=np.float64)
    pixmap = np.array([[[1.25, 2.5]]])

    result = blot(10.0 * x + y, pixmap)

    assert result[0, 0] == pytest.approx(15.0, abs=1e-12)


def test_blot_last_column_interpolates_along_y():
    """blot of a 3x4 image at x 3.5 (in its last column), y 0.25 gives 0.75 of row 0 plus 0.25 of row 1 in that column."""
    data = np.arange(12.0).reshape(3, 4)
    pixmap = np.array([[[3.5, 0.25]]])

    result = blot(data, pixmap)

    assert result[0, 0] == pytest.approx(0.75 * 3.0 + 0.25 * 7.0, abs=1e-12)


def test_blot_last_row_interpolates_along_x():
    """blot of a 3x4 image at x 1.25, y 2.5 (in its last row) gives 0.75 of column 1 plus 0.25 of column 2 in that row."""
    data = np.arange(12.0).reshape(3, 4)
    pixmap = np.array([[[1.25, 2.5]]])

    result = blot(data, pixmap)

    assert result[0, 0] == pytest.approx(0.75 * 9.0 + 0.25 * 10.0, abs=1e-12)


def test_blot_last_pixel_is_its_value():
    """blot of a 3x4 image at (3.5, 2.5), in its last pixel, gives that pixel's value."""
    data = np.arange(12.0).reshape(3, 4)
    pixmap = np.array([[[3.5, 2.5]]])

    result = blot(data, pixmap)

    assert result[0, 0] == 11.0


@pytest.mark.parametrize("position", [(-0.1, 1.0), (4.0, 1.0), (1.0, -0.1), (1.0, 3.0), (np.nan, 1.0)])
def test_blot_fills_positions_outside_the_image(position):
    """blot of a 3x4 image at a position outside [0, 4) x [0, 3), or NaN, gives the fill value."""
    data = np.arange(12.0).reshape(3, 4)
    pixmap = np.array([[position]])

    result = blot(data, pixmap, fillval=-7.0)

    assert result[0, 0] == -7.0


def test_blot_same_on_one_thread():
    """blot of a random image on one numba thread gives the same result as on all threads."""
    data = np.random.default_rng(1).normal(size=DATA_SHAPE)
    pixmap = random_pixmap()
    on_all = blot(data, pixmap)

    numba.set_num_threads(1)
    try:
        on_one = blot(data, pixmap)
    finally:
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    np.testing.assert_array_equal(on_one, on_all)


def test_blot_rejects_out_of_the_wrong_shape():
    """blot with out of shape (40, 44) for a (40, 45) pixel map raises ValueError."""
    with pytest.raises(ValueError, match="out"):
        blot(np.zeros(DATA_SHAPE), random_pixmap(), out=np.zeros((40, 44)))
