"""Comparison with drizzle's square kernel.

drizzle must accept float64 output arrays, from
jdavies-st/drizzle@float64-phase2-c-accepts-float64 until that is released.
It runs its running weighted mean in double precision, so the results
agree to float64 round-off.  The tolerances allow for drizzle builds that
contract multiplications and additions into fused multiply-adds.
"""
import numpy as np
import pytest
from drizzle import cdrizzle

from spizzle import Coadd, drizzle_operator

SHAPE = (30, 40)
OUT_SHAPE = (130, 140)
CENTER = (70.0, 65.0)


def drizzle_outputs(out_shape):
    """Float64 output arrays for cdrizzle.tdriz, NaN where nothing lands, as drizzle.resample.Drizzle starts them."""
    return {
        "output": np.full(out_shape, np.nan),
        "counts": np.zeros(out_shape),
        "context": np.zeros(out_shape, dtype=np.int32),
        "output2": None,
    }


def drizzle_outputs_with_variance(out_shape, n_var):
    """drizzle_outputs with ``n_var`` float64 variance output arrays."""
    outputs = drizzle_outputs(out_shape)
    outputs["output2"] = [np.full(out_shape, np.nan) for _ in range(n_var)]
    return outputs


def tdriz(outputs, data, weights, pixmap, pixfrac=1.0, variances=None, uniqid=1, iscale=1.0, wtscale=1.0, **limits):
    """Drizzle one image into ``outputs`` with drizzle's square kernel; ``limits`` are tdriz's xmin, xmax, ymin, ymax."""
    cdrizzle.tdriz(
        input=data,
        weights=weights,
        pixmap=pixmap,
        output=outputs["output"],
        counts=outputs["counts"],
        context=outputs["context"],
        input2=variances,
        output2=outputs["output2"],
        uniqid=uniqid,
        iscale=iscale,
        pixfrac=pixfrac,
        kernel="square",
        in_units="cps",
        expscale=1.0,
        wtscale=wtscale,
        # tdriz takes the fill value as a string
        fillstr=str(np.nan),
        **limits,
    )


def _require_float64_drizzle():
    """Raise ImportError unless cdrizzle.tdriz writes float64 output arrays."""
    y, x = np.indices((3, 3), dtype=np.float64)
    try:
        tdriz(drizzle_outputs((3, 3)), np.ones((3, 3)), np.ones((3, 3)), np.dstack([x, y]))
    except (TypeError, ValueError) as err:
        raise ImportError(
            "the drizzle comparison tests need drizzle with float64 output arrays: pip install "
            "'drizzle @ git+https://github.com/jdavies-st/drizzle.git@float64-phase2-c-accepts-float64'"
        ) from err


_require_float64_drizzle()


def distorted_pixmap(shape, degrees, scale, center):
    """Pixel map of an input with quadratic distortion, rotated by ``degrees``, scaled by ``scale`` and centered at ``center``."""
    t = np.deg2rad(degrees)
    y, x = np.indices(shape, dtype=np.float64)
    xc = x - shape[1] / 2
    yc = y - shape[0] / 2
    xd = xc + 2e-3 * xc * yc
    yd = yc + 1e-3 * xc * xc
    xo = scale * (np.cos(t) * xd - np.sin(t) * yd) + center[0]
    yo = scale * (np.sin(t) * xd + np.cos(t) * yd) + center[1]
    return np.dstack([xo, yo])


def random_image(shape, low, high, seed):
    """Uniform random image between ``low`` and ``high``."""
    return np.random.default_rng(seed).uniform(low, high, shape)


@pytest.mark.parametrize("pixel", [(0, 0), (0, 39), (29, 0), (29, 39), (1, 1), (28, 38), (15, 20)])
def test_operator_column_matches_drizzle_weights_of_one_pixel(pixel):
    """drizzle_operator's column for one input pixel equals drizzle's output weights with only that pixel weighted, to 1e-12."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    weights = np.zeros(SHAPE)
    weights[pixel] = 1.0
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, np.ones(SHAPE), weights, pixmap, pixfrac=0.7)

    W = drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.7).full()
    column = W[:, [np.ravel_multi_index(pixel, SHAPE)]].toarray()

    np.testing.assert_allclose(column.reshape(OUT_SHAPE), outputs["counts"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("pixfrac", [0.5, 0.8, 1.0])
def test_weights_match_drizzle_for_pixfrac(pixfrac):
    """Coadd.out_wht of a distorted, weighted exposure equals drizzle's output weights for the same pixfrac."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    weights = random_image(SHAPE, 0.5, 1.5, seed=1)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, np.ones(SHAPE), weights, pixmap, pixfrac=pixfrac)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=pixfrac, weights=weights), np.ones(SHAPE))

    np.testing.assert_allclose(coadd.out_wht, outputs["counts"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pixfrac", [0.5, 0.8, 1.0])
def test_science_matches_drizzle_for_pixfrac(pixfrac):
    """Coadd.out_img of a distorted, weighted exposure equals drizzle's output, NaN in the same pixels, for the same pixfrac."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    data = random_image(SHAPE, 10.0, 11.0, seed=2)
    weights = random_image(SHAPE, 0.5, 1.5, seed=3)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, data, weights, pixmap, pixfrac=pixfrac)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=pixfrac, weights=weights), data)

    np.testing.assert_allclose(coadd.out_img, outputs["output"], rtol=1e-12)


def test_region_matches_drizzle_limits():
    """Coadd.out_img with region rows 5-20, columns 8-30 equals tdriz's output for ymin 5, ymax 20, xmin 8, xmax 30."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    data = random_image(SHAPE, 10.0, 11.0, seed=5)
    weights = random_image(SHAPE, 0.5, 1.5, seed=6)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, data, weights, pixmap, pixfrac=0.8, xmin=8, xmax=30, ymin=5, ymax=20)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8, weights=weights, region=(slice(5, 21), slice(8, 31))), data)

    np.testing.assert_allclose(coadd.out_img, outputs["output"], rtol=1e-12)


def test_region_weights_match_drizzle_limits():
    """Coadd.out_wht with region rows 5-20, columns 8-30 equals tdriz's weights for ymin 5, ymax 20, xmin 8, xmax 30."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    weights = random_image(SHAPE, 0.5, 1.5, seed=6)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, np.ones(SHAPE), weights, pixmap, pixfrac=0.8, xmin=8, xmax=30, ymin=5, ymax=20)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8, weights=weights, region=(slice(5, 21), slice(8, 31))),
              np.ones(SHAPE))

    np.testing.assert_allclose(coadd.out_wht, outputs["counts"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("degrees", [0.0, 30.0, 90.0, 137.0])
def test_science_matches_drizzle_for_rotation(degrees):
    """Coadd.out_img of an exposure rotated by ``degrees`` equals drizzle's output."""
    pixmap = distorted_pixmap(SHAPE, degrees, 1.7, CENTER)
    data = random_image(SHAPE, 10.0, 11.0, seed=4)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, data, np.ones(SHAPE), pixmap, pixfrac=0.8)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8), data)

    np.testing.assert_allclose(coadd.out_img, outputs["output"], rtol=1e-12)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_science_matches_drizzle_for_scale(scale):
    """Coadd.out_img of an exposure whose pixels are ``scale`` output pixels across equals drizzle's output."""
    pixmap = distorted_pixmap(SHAPE, 23.0, scale, CENTER)
    data = random_image(SHAPE, 10.0, 11.0, seed=5)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, data, np.ones(SHAPE), pixmap, pixfrac=0.8)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8), data)

    np.testing.assert_allclose(coadd.out_img, outputs["output"], rtol=1e-12)


def test_science_matches_drizzle_for_flipped_input():
    """Coadd.out_img of an exposure mirrored in x, with negative jacobian, equals drizzle's output."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    pixmap[..., 0] = 2 * CENTER[0] - pixmap[..., 0]
    data = random_image(SHAPE, 10.0, 11.0, seed=6)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, data, np.ones(SHAPE), pixmap, pixfrac=0.8)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8), data)

    np.testing.assert_allclose(coadd.out_img, outputs["output"], rtol=1e-12)


def test_science_matches_drizzle_with_nan_pixels_of_zero_weight():
    """Coadd.out_img of an exposure with 40 NaN pixels of weight 0 equals drizzle's output, which has no NaN from them."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    data = random_image(SHAPE, 10.0, 11.0, seed=7)
    weights = random_image(SHAPE, 0.5, 1.5, seed=8)
    bad = np.random.default_rng(9).choice(data.size, 40, replace=False)
    data.flat[bad] = np.nan
    weights.flat[bad] = 0.0
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, data, weights, pixmap, pixfrac=0.8)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8, weights=weights), data)

    np.testing.assert_allclose(coadd.out_img, outputs["output"], rtol=1e-12)


def test_variance_matches_drizzle():
    """Coadd.out_var of an exposure with two variance planes equals drizzle's output2 arrays."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    data = random_image(SHAPE, 10.0, 11.0, seed=10)
    weights = random_image(SHAPE, 0.5, 1.5, seed=11)
    variances = [random_image(SHAPE, 1.0, 2.0, seed=12), random_image(SHAPE, 3.0, 4.0, seed=13)]
    outputs = drizzle_outputs_with_variance(OUT_SHAPE, 2)
    tdriz(outputs, data, weights, pixmap, pixfrac=0.8, variances=variances)
    coadd = Coadd(OUT_SHAPE, n_var=2)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8, weights=weights), data, variances=variances)

    np.testing.assert_allclose(np.array(coadd.out_var), np.array(outputs["output2"]), rtol=1e-12)


def test_science_with_iscale_matches_drizzle():
    """Coadd.out_img of an exposure added with iscale 2.5 equals drizzle's output with iscale 2.5."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    data = random_image(SHAPE, 10.0, 11.0, seed=14)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, data, np.ones(SHAPE), pixmap, pixfrac=0.8, iscale=2.5)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8), data, iscale=2.5)

    np.testing.assert_allclose(coadd.out_img, outputs["output"], rtol=1e-12)


def test_variance_with_iscale_matches_drizzle():
    """Coadd.out_var of an exposure added with iscale 2.5 equals drizzle's output2 with iscale 2.5."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    variances = [random_image(SHAPE, 1.0, 2.0, seed=15)]
    outputs = drizzle_outputs_with_variance(OUT_SHAPE, 1)
    tdriz(outputs, np.ones(SHAPE), np.ones(SHAPE), pixmap, pixfrac=0.8, variances=variances, iscale=2.5)
    coadd = Coadd(OUT_SHAPE, n_var=1)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8), np.ones(SHAPE), variances=variances, iscale=2.5)

    np.testing.assert_allclose(coadd.out_var[0], outputs["output2"][0], rtol=1e-12)


def test_weights_with_weight_scale_match_drizzle():
    """Coadd.out_wht of an exposure with weight_scale 0.3 equals drizzle's output weights with wtscale 0.3."""
    pixmap = distorted_pixmap(SHAPE, 23.0, 1.7, CENTER)
    weights = random_image(SHAPE, 0.5, 1.5, seed=16)
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, np.ones(SHAPE), weights, pixmap, pixfrac=0.8, wtscale=0.3)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmap, OUT_SHAPE, pixfrac=0.8, weights=weights, weight_scale=0.3), np.ones(SHAPE))

    np.testing.assert_allclose(coadd.out_wht, outputs["counts"], rtol=1e-12, atol=1e-14)


def test_two_exposures_match_drizzle():
    """Coadd.out_img of two overlapping exposures with different rotations equals drizzle's output after both."""
    pixmaps = [distorted_pixmap(SHAPE, 23.0, 1.7, CENTER), distorted_pixmap(SHAPE, 61.0, 1.7, (75.0, 60.0))]
    data = [random_image(SHAPE, 10.0, 11.0, seed=17), random_image(SHAPE, 12.0, 13.0, seed=18)]
    weights = [random_image(SHAPE, 0.5, 1.5, seed=19), random_image(SHAPE, 0.5, 1.5, seed=20)]
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, data[0], weights[0], pixmaps[0], pixfrac=0.8, uniqid=1)
    tdriz(outputs, data[1], weights[1], pixmaps[1], pixfrac=0.8, uniqid=2)
    coadd = Coadd(OUT_SHAPE)

    coadd.add(drizzle_operator(pixmaps[0], OUT_SHAPE, pixfrac=0.8, weights=weights[0]), data[0])
    coadd.add(drizzle_operator(pixmaps[1], OUT_SHAPE, pixfrac=0.8, weights=weights[1]), data[1])

    np.testing.assert_allclose(coadd.out_img, outputs["output"], rtol=1e-12)


def test_context_of_two_exposures_matches_drizzle():
    """Coadd.out_ctx of two overlapping exposures equals drizzle's context with uniqid 1 and 2."""
    pixmaps = [distorted_pixmap(SHAPE, 23.0, 1.7, CENTER), distorted_pixmap(SHAPE, 61.0, 1.7, (75.0, 60.0))]
    outputs = drizzle_outputs(OUT_SHAPE)
    tdriz(outputs, np.ones(SHAPE), np.ones(SHAPE), pixmaps[0], pixfrac=0.8, uniqid=1)
    tdriz(outputs, np.ones(SHAPE), np.ones(SHAPE), pixmaps[1], pixfrac=0.8, uniqid=2)
    coadd = Coadd(OUT_SHAPE, context=True)

    coadd.add(drizzle_operator(pixmaps[0], OUT_SHAPE, pixfrac=0.8), np.ones(SHAPE))
    coadd.add(drizzle_operator(pixmaps[1], OUT_SHAPE, pixfrac=0.8), np.ones(SHAPE))

    np.testing.assert_array_equal(coadd.out_ctx[0], outputs["context"])


def test_partial_overlap_matches_drizzle_away_from_the_edges():
    """Coadd.out_img of an exposure larger than the output equals drizzle's output except in the outer two rings.

    drizzle skips input pixels centered outside the output frame, so the outer
    two rings of output pixels get less weight from drizzle than from spizzle.
    """
    pixmap = distorted_pixmap(SHAPE, 30.0, 1.0, (14.0, 12.0))
    data = random_image(SHAPE, 10.0, 11.0, seed=21)
    weights = random_image(SHAPE, 0.5, 1.5, seed=22)
    outputs = drizzle_outputs((24, 28))
    tdriz(outputs, data, weights, pixmap, pixfrac=0.8)
    coadd = Coadd((24, 28))

    coadd.add(drizzle_operator(pixmap, (24, 28), pixfrac=0.8, weights=weights), data)

    np.testing.assert_allclose(coadd.out_img[2:-2, 2:-2], outputs["output"][2:-2, 2:-2], rtol=1e-12)


def test_output_edge_weight_differs_from_drizzle():
    """For an input extending 9.7 pixels past the left output edge, output column 0 gets weight 1 from spizzle, 0.7 from drizzle.

    This test fails once drizzle stops skipping input pixels centered outside
    the output frame; spizzle's weights are then drizzle's.
    """
    y, x = np.indices((12, 20), dtype=np.float64)
    pixmap = np.dstack([x - 9.7, y])
    outputs = drizzle_outputs((12, 10))
    tdriz(outputs, np.ones((12, 20)), np.ones((12, 20)), pixmap)
    coadd = Coadd((12, 10))

    coadd.add(drizzle_operator(pixmap, (12, 10)), np.ones((12, 20)))

    np.testing.assert_allclose(coadd.out_wht[:, 0], np.ones(12), rtol=1e-14)
    np.testing.assert_allclose(outputs["counts"][:, 0], np.full(12, 0.7), rtol=1e-14)
