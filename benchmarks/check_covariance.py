"""
Check spizzle's Covariance against noise realizations and the drizzle noise correction.

Run from the repository root::

    python benchmarks/check_covariance.py [DATA/jw*_nrcalong_cal.fits]

1. Monte Carlo: a rotated, scaled exposure with random weights and
   variances is drizzled with ``--realizations`` noise realizations
   through `Coadd`.  For each offset the difference between the sample
   covariance and `Covariance` is divided by its statistical error,
   ``sqrt((var_o var_o' + cov^2) / n)``; over the pixels it should have
   mean ~0 and standard deviation ~1.
2. Casertano et al. (2000, AJ 120, 2747): for uniform weights and noise,
   the noise of a sum over a large area exceeds the one computed from the
   pixel variance by a factor R, which they give as ``1 / (1 - r / 3)``
   for ``r <= 1`` and ``r / (1 - 1 / (3 r))`` for ``r >= 1``, with
   ``r = pixfrac / s`` and ``s`` the output pixel size in input pixels.
   The formula assumes dithers that cover the output evenly; here 16
   exposures at random sub-pixel offsets, slightly rotated, are added to
   one `Covariance`, and ``R**2`` is the sum of the covariance over all
   offsets over the variance, averaged over the interior.
3. With JWST ``cal`` files (and gwcs), the time of `Covariance.add` for
   one exposure on a mosaic frame enclosing them all.
"""

import argparse
import time
import warnings

import numpy as np

from spizzle import Coadd, Covariance, drizzle_operator


def rotated_pixmap(shape, degrees, scale, center):
    """Pixel map of an input rotated by ``degrees`` and scaled by ``scale``, its center at ``center`` (x, y)."""
    t = np.deg2rad(degrees)
    y, x = np.indices(shape, dtype=np.float64)
    xc = x - shape[1] / 2
    yc = y - shape[0] / 2
    xo = scale * (np.cos(t) * xc - np.sin(t) * yc) + center[0]
    yo = scale * (np.sin(t) * xc + np.cos(t) * yc) + center[1]
    return np.dstack([xo, yo])


def shifted(plane, dy, dx):
    """Views of ``plane`` at (y, x) and at (y + dy, x + dx), for the pixels where both are on the grid."""
    ny, nx = plane.shape
    here = (slice(0, ny - dy), slice(max(0, -dx), nx - max(0, dx)))
    there = (slice(dy, ny), slice(max(0, dx), nx - max(0, -dx)))
    return plane[here], plane[there]


def monte_carlo(n_real, radius, seed=1):
    shape, out_shape = (60, 80), (150, 160)
    rng = np.random.default_rng(seed)
    op = drizzle_operator(rotated_pixmap(shape, 33.0, 1.6, (80.0, 75.0)), out_shape, pixfrac=0.8,
                          weights=rng.uniform(0.5, 1.5, shape))
    var = rng.uniform(1.0, 4.0, shape)
    cov = Covariance(out_shape, radius=radius)
    cov.add(op, var)
    predicted = cov.out_cov

    sums = np.zeros(out_shape)
    products = np.zeros((len(cov.offsets),) + out_shape)
    for _ in range(n_real):
        coadd = Coadd(out_shape, fillval=0.0)
        coadd.add(op, rng.normal(0.0, np.sqrt(var)))
        img = coadd.out_img
        sums += img
        for k, (dy, dx) in enumerate(cov.offsets):
            a, b = shifted(img, dy, dx)
            shifted(products[k], dy, dx)[0][...] += a * b

    mean = sums / n_real
    print(f"\nMonte Carlo, {n_real} realizations of a 60x80 exposure (rotated 33 deg, 1.6x, pixfrac 0.8) on 150x160:\n")
    print("| offset (dy, dx) | pixels | mean z | std z |")
    print("| --- | ---: | ---: | ---: |")
    var_out = predicted[0]
    for k, (dy, dx) in enumerate(cov.offsets):
        pred_here, _ = shifted(predicted[k], dy, dx)
        var_a, var_b = shifted(var_out, dy, dx)
        mean_a, mean_b = shifted(mean, dy, dx)
        sample = shifted(products[k], dy, dx)[0] / n_real - mean_a * mean_b
        error = np.sqrt((var_a * var_b + pred_here ** 2) / n_real)
        good = np.isfinite(pred_here) & (error > 0)
        z = (sample[good] - pred_here[good]) / error[good]
        print(f"| ({dy}, {dx}) | {good.sum()} | {z.mean():+.3f} | {z.std():.3f} |")


def casertano(r):
    """Casertano et al. (2000) noise ratio for r = pixfrac / s."""
    return 1.0 / (1.0 - r / 3.0) if r <= 1.0 else r / (1.0 - 1.0 / (3.0 * r))


def noise_ratio(pixfrac, s, radius, n_dither=16, seed=2):
    """sqrt of (sum of the covariance over all offsets) / variance, over the interior of dithered exposures."""
    shape = (200, 200)
    scale = 1.0 / s  # input pixel size in output pixels
    out_n = int(np.ceil(200 * scale * 1.1)) + 8
    out_shape = (out_n, out_n)
    cov = Covariance(out_shape, radius=radius)
    for dx, dy in np.random.default_rng(seed).uniform(0.0, 1.0, (n_dither, 2)):
        center = (out_n / 2 + dx * scale, out_n / 2 + dy * scale)
        cov.add(drizzle_operator(rotated_pixmap(shape, 1.3, scale, center), out_shape, pixfrac=pixfrac), np.ones(shape))
    planes = cov.out_cov
    # the interior, well inside the footprint of the input
    m = out_n // 4
    inner = (slice(m, out_n - m), slice(m, out_n - m))
    var = planes[0][inner].mean()
    total = var + 2.0 * sum(planes[k][inner].mean() for k in range(1, len(cov.offsets)))
    return np.sqrt(total / var)


def real_data(files, radius):
    import sys
    sys.path.insert(0, "benchmarks")
    from compare_jwst import Exposure, output_frame
    from spizzle import pixel_map

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exposures = [Exposure(path, "ivm") for path in files]
    print(f"\nNIRCam, {len(exposures)} exposures, radius {radius}:\n")
    print("| frame | output | Covariance.add per exposure | memory of the planes |")
    print("| --- | ---: | ---: | ---: |")
    for ratio, pixfrac in [(1.0, 1.0), (0.5, 0.8)]:
        wcs_out, out_shape = output_frame(exposures, ratio)
        e = exposures[0]
        op = drizzle_operator(pixel_map(e.wcs, wcs_out, shape=e.data.shape), out_shape, pixfrac=pixfrac, weights=e.weights)
        var = sum(e.variances)
        cov = Covariance(out_shape, radius=radius)
        cov.add(op, var)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            cov.add(op, var)
            times.append(time.perf_counter() - t0)
        size = cov._cnum.nbytes / 2**30
        print(f"| {ratio}x scale, pixfrac {pixfrac} | {out_shape[0]}x{out_shape[1]} | {min(times):.3f} s | {size:.1f} GiB |")
        del cov, op


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("files", nargs="*", help="JWST cal files for the timing")
    parser.add_argument("--realizations", type=int, default=4000)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    monte_carlo(args.realizations, args.radius)

    print("\nNoise ratio R for large areas, uniform noise, 16 dithers:\n")
    print("| pixfrac | s | r = pixfrac / s | R from Covariance | Casertano et al. |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for pixfrac in (0.5, 0.75, 1.0):
        for s in (0.5, 0.75, 1.0):
            r = pixfrac / s
            print(f"| {pixfrac} | {s} | {r:.3f} | {noise_ratio(pixfrac, s, radius=4):.4f} | {casertano(r):.4f} |", flush=True)

    if args.files:
        real_data(args.files, args.radius)


if __name__ == "__main__":
    main()
