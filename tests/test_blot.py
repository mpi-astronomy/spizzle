"""Comparison with drizzle's bilinear blot.

drizzle must accept float64 arrays in ``tblot``, from
jdavies-st/drizzle@float64-phase2-c-accepts-float64 until that is
released.  The tolerances allow for drizzle builds that contract
multiplications and additions into fused multiply-adds.  blot accepts the
image's extent, ``[-0.5, n - 0.5)``, where drizzle accepts ``[0, n)``, so
the comparisons use positions where the two agree.
"""
import numba
import numpy as np
import pytest
from drizzle import cdrizzle

from spizzle import blot

DATA_SHAPE = (50, 60)
SHAPE = (40, 45)


def drizzle_pixmap(seed=2):
    """Positions in a 50x60 image where blot and drizzle accept the same: inside [0, 59.5) x [0, 49.5), or outside both.

    Includes the last row, column and pixel, positions beyond each edge,
    and NaN.
    """
    rng = np.random.default_rng(seed)
    pixmap = np.dstack([rng.uniform(0.0, 59.5, SHAPE), rng.uniform(0.0, 49.5, SHAPE)])
    pixmap[0, :5, 0] = 59.0 + rng.uniform(0.0, 0.5, 5)
    pixmap[1, :5, 1] = 49.0 + rng.uniform(0.0, 0.5, 5)
    pixmap[2, 0] = (59.25, 49.25)
    pixmap[3, :10, 0] = rng.uniform(-2.0, -0.5, 10)
    pixmap[4, :10, 0] = rng.uniform(60.0, 62.0, 10)
    pixmap[5, :10, 1] = rng.uniform(-2.0, -0.5, 10)
    pixmap[6, :10, 1] = rng.uniform(50.0, 52.0, 10)
    pixmap[7, :3, 0] = np.nan
    return pixmap


def tblot(data, pixmap, out, iscale=1.0, fillval=np.nan):
    """drizzle's bilinear blot of ``data`` into ``out``."""
    cdrizzle.tblot(data, pixmap, out, iscale=iscale, interp="linear", fillval=fillval)
    return out


def small_image():
    """The 3x4 image with rows [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]."""
    return np.arange(1.0, 13.0).reshape(3, 4)


def test_blot_matches_drizzle():
    """blot of a random 50x60 image, with iscale 1.7, gives drizzle's tblot result to 1e-12 where both accept the positions."""
    data = np.random.default_rng(1).normal(size=DATA_SHAPE)
    pixmap = drizzle_pixmap()
    expected = tblot(data, pixmap, np.zeros(SHAPE), iscale=1.7, fillval=-5.0)

    result = blot(data, pixmap, fillval=-5.0, iscale=1.7)

    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_blot_float32_matches_drizzle():
    """blot of a float32 image into a float32 array gives drizzle's tblot result to 1 float32 rounding step."""
    data = np.random.default_rng(1).normal(size=DATA_SHAPE).astype(np.float32)
    pixmap = drizzle_pixmap()
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
    """blot of the 3x4 image at x 3.25 (beyond its last column's centers), y 0.25 gives 0.75 * 4 + 0.25 * 8 = 5."""
    pixmap = np.array([[[3.25, 0.25]]])

    result = blot(small_image(), pixmap)

    assert result[0, 0] == pytest.approx(5.0, abs=1e-12)


def test_blot_first_column_interpolates_along_y():
    """blot of the 3x4 image at x -0.25 (before its first column's centers), y 0.25 gives 0.75 * 1 + 0.25 * 5 = 2."""
    pixmap = np.array([[[-0.25, 0.25]]])

    result = blot(small_image(), pixmap)

    assert result[0, 0] == pytest.approx(2.0, abs=1e-12)


def test_blot_last_row_interpolates_along_x():
    """blot of the 3x4 image at x 1.25, y 2.25 (beyond its last row's centers) gives 0.75 * 10 + 0.25 * 11 = 10.25."""
    pixmap = np.array([[[1.25, 2.25]]])

    result = blot(small_image(), pixmap)

    assert result[0, 0] == pytest.approx(10.25, abs=1e-12)


def test_blot_first_row_interpolates_along_x():
    """blot of the 3x4 image at x 1.25, y -0.25 (before its first row's centers) gives 0.75 * 2 + 0.25 * 3 = 2.25."""
    pixmap = np.array([[[1.25, -0.25]]])

    result = blot(small_image(), pixmap)

    assert result[0, 0] == pytest.approx(2.25, abs=1e-12)


def test_blot_last_pixel_is_its_value():
    """blot of the 3x4 image at (3.25, 2.25), in the outer quarter of its last pixel, gives that pixel's value, 12."""
    pixmap = np.array([[[3.25, 2.25]]])

    result = blot(small_image(), pixmap)

    assert result[0, 0] == 12.0


def test_blot_first_pixel_is_its_value():
    """blot of the 3x4 image at (-0.25, -0.25), in the outer quarter of its first pixel, gives that pixel's value, 1."""
    pixmap = np.array([[[-0.25, -0.25]]])

    result = blot(small_image(), pixmap)

    assert result[0, 0] == 1.0


@pytest.mark.parametrize("position", [(-0.6, 1.0), (3.5, 1.0), (1.0, -0.6), (1.0, 2.5), (np.nan, 1.0)])
def test_blot_fills_positions_outside_the_image(position):
    """blot of the 3x4 image at a position outside its extent [-0.5, 3.5) x [-0.5, 2.5), or NaN, gives the fill value."""
    pixmap = np.array([[position]])

    result = blot(small_image(), pixmap, fillval=-7.0)

    assert result[0, 0] == -7.0


def test_blot_same_on_one_thread():
    """blot of a random image on one numba thread gives the same result as on all threads."""
    data = np.random.default_rng(1).normal(size=DATA_SHAPE)
    pixmap = drizzle_pixmap()
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
        blot(np.zeros(DATA_SHAPE), drizzle_pixmap(), out=np.zeros((40, 44)))
