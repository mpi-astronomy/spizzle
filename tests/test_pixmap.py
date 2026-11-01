import numpy as np
import pytest
from drizzle.utils import calc_pixmap
from scipy.interpolate import RectBivariateSpline

from spizzle import pixel_map
from spizzle.pixmap import _spline_on_grid

SHAPE = (200, 300)


class InputWCS:
    """Input WCS with a smooth distortion, rotated and scaled onto the world plane.

    ``pixel_to_world_values`` gives NaN outside ``bounding_box`` (``((xmin,
    xmax), (ymin, ymax))``), as gwcs does, and inside the disc ``hole``
    (``(x, y, radius)``), which has no mapping.
    """

    def __init__(self, distortion=1.0, bounding_box=None, hole=None):
        self.array_shape = SHAPE
        self.bounding_box = bounding_box
        self.distortion = distortion
        self.hole = hole

    def pixel_to_world_values(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xd = x + self.distortion * (2e-5 * (x - 100) * (y - 80) + 0.3 * np.sin(x / 150))
        yd = y + self.distortion * (1e-5 * x * x + 0.2 * np.cos(y / 120))
        t = np.deg2rad(20.0)
        a = 0.9 * (np.cos(t) * xd - np.sin(t) * yd)
        b = 0.9 * (np.sin(t) * xd + np.cos(t) * yd)
        outside = np.zeros(x.shape, dtype=bool)
        if self.bounding_box is not None:
            (xlo, xhi), (ylo, yhi) = self.bounding_box
            outside |= (x < xlo) | (x > xhi) | (y < ylo) | (y > yhi)
        if self.hole is not None:
            hx, hy, r = self.hole
            outside |= (x - hx) ** 2 + (y - hy) ** 2 < r * r
        return np.where(outside, np.nan, a), np.where(outside, np.nan, b)


class OutputWCS:
    """Output WCS: the world plane shifted by (120, 30) world units."""

    def world_to_pixel_values(self, a, b):
        return np.asarray(a) + 120.0, np.asarray(b) + 30.0


def test_pixel_map_of_every_pixel_matches_calc_pixmap():
    """pixel_map with stepsize 1 of a distorted input gives drizzle's calc_pixmap result bit for bit."""
    expected = calc_pixmap(InputWCS(), OutputWCS(), shape=SHAPE)

    pixmap = pixel_map(InputWCS(), OutputWCS(), shape=SHAPE, stepsize=1)

    np.testing.assert_array_equal(pixmap, expected)


def test_pixel_map_of_every_pixel_same_on_one_thread():
    """pixel_map with stepsize 1 of a distorted input on one thread gives the same result as on four."""
    on_four = pixel_map(InputWCS(), OutputWCS(), shape=SHAPE, stepsize=1, threads=4)

    on_one = pixel_map(InputWCS(), OutputWCS(), shape=SHAPE, stepsize=1, threads=1)

    np.testing.assert_array_equal(on_one, on_four)


def test_pixel_map_interpolated_matches_every_pixel():
    """pixel_map with the default bicubic splines every 16 pixels of a distorted input is within 1e-5 pixel of calc_pixmap."""
    expected = calc_pixmap(InputWCS(), OutputWCS(), shape=SHAPE)

    pixmap = pixel_map(InputWCS(), OutputWCS(), shape=SHAPE)

    np.testing.assert_allclose(pixmap, expected, rtol=0, atol=1e-5)


def test_pixel_map_bilinear_of_undistorted_input():
    """pixel_map with bilinear interpolation of an input without distortion gives calc_pixmap's result to 1e-10 pixel."""
    expected = calc_pixmap(InputWCS(distortion=0.0), OutputWCS(), shape=SHAPE)

    pixmap = pixel_map(InputWCS(distortion=0.0), OutputWCS(), shape=SHAPE, order=1)

    np.testing.assert_allclose(pixmap, expected, rtol=0, atol=1e-10)


def test_pixel_map_is_nan_outside_the_input_bounding_box():
    """pixel_map of an input whose bounding box covers columns 0-99 and rows 0-149 is NaN in the other pixels only."""
    y, x = np.indices(SHAPE)
    expected = (x > 99) | (y > 149)
    wcs = InputWCS(bounding_box=((-0.5, 99.5), (-0.5, 149.5)))

    pixmap = pixel_map(wcs, OutputWCS(), shape=SHAPE)

    np.testing.assert_array_equal(np.isnan(pixmap).any(axis=2), expected)


def test_pixel_map_of_every_pixel_with_bounding_box_matches_calc_pixmap():
    """pixel_map with stepsize 1 of an input with a bounding box gives calc_pixmap's result, NaN in the same pixels."""
    wcs = InputWCS(bounding_box=((-0.5, 99.5), (-0.5, 149.5)))
    expected = calc_pixmap(wcs, OutputWCS(), shape=SHAPE)

    pixmap = pixel_map(wcs, OutputWCS(), shape=SHAPE, stepsize=1)

    np.testing.assert_array_equal(pixmap, expected)


def test_pixel_map_evaluates_every_pixel_when_the_grid_has_nan():
    """pixel_map of an input with a hole of radius 20 at (150, 100) gives calc_pixmap's result, NaN in the hole."""
    wcs = InputWCS(hole=(150.0, 100.0, 20.0))
    expected = calc_pixmap(wcs, OutputWCS(), shape=SHAPE)

    pixmap = pixel_map(wcs, OutputWCS(), shape=SHAPE)

    np.testing.assert_array_equal(pixmap, expected)


def test_pixel_map_takes_the_shape_from_the_input_wcs():
    """pixel_map without shape of an input WCS with array_shape (200, 300) gives a pixel map of shape (200, 300, 2)."""
    pixmap = pixel_map(InputWCS(), OutputWCS())

    assert pixmap.shape == (200, 300, 2)


def test_spline_on_grid_matches_scipy():
    """_spline_on_grid of a bicubic spline through a 12x15 grid gives scipy's RectBivariateSpline values to 1e-12."""
    y_grid = np.linspace(0.0, 110.0, 12)
    x_grid = np.linspace(0.0, 140.0, 15)
    values = np.random.default_rng(1).uniform(-1.0, 1.0, (12, 15))
    y_all = np.arange(111.0)
    x_all = np.arange(141.0)
    expected = RectBivariateSpline(y_grid, x_grid, values, kx=3, ky=3)(y_all, x_all)

    result = _spline_on_grid(y_grid, x_grid, values, 3, y_all, x_all)

    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_pixel_map_rejects_order_2():
    """pixel_map with order 2 raises ValueError."""
    with pytest.raises(ValueError, match="order"):
        pixel_map(InputWCS(), OutputWCS(), shape=SHAPE, order=2)
