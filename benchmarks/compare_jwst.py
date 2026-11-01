"""
Compare spizzle with drizzle on JWST imaging exposures, and time both.

Run from the repository root with a set of ``cal`` or ``crf`` files::

    python benchmarks/compare_jwst.py DATA/jw*_nrcalong_cal.fits

The exposures are drizzled onto one north-up tangent-plane output frame
that encloses them all, by drizzle (``cdrizzle.tdriz``, square kernel, with
float32 and with float64 output arrays) and by spizzle, from the same pixel
maps, weights and variance planes.  The pixel maps come from
``drizzle.utils.calc_pixmap``, as in the jwst pipeline.  The weights are
those of the pipeline's ``weight_type``: the inverse read-noise variance
(``ivm``) or the exposure time (``exptime``), zero where the DQ array has
DO_NOT_USE or NON_SCIENCE set.  The science data, weights and the
VAR_POISSON, VAR_RNOISE and VAR_FLAT planes go into one ``tdriz`` call per
exposure, as jwst passes them.

For each output plane the script prints how many output pixels differ
from drizzle by how much, in decades of the difference relative to
``max(|drizzle|, median |drizzle|)`` over the pixels with weight, so that
science values close to zero do not dominate.  It also compares the NaN
pixels and the context image, checks that the flux is conserved, and
times the pixel maps (gwcs), the spizzle operator and co-addition and the
drizzle calls.  It also times spizzle's ``pixel_map``, which interpolates
the WCS evaluated on a coarse grid, and prints its largest difference from
``calc_pixmap``.  The packages run ``--repeat`` times over all exposures, in
alternating order and with only one package's outputs in memory at a
time, and the timings are the minimum for each exposure; the outputs
compared come from one more, untimed, pass of each.  The pixel maps are
computed once.

Reading the WCS needs gwcs (``pip install gwcs``); jwst is not needed.
"""

import argparse
import io
import time
import warnings

import asdf
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from drizzle import cdrizzle
from drizzle.utils import calc_pixmap

from spizzle import Coadd, drizzle_operator, pixel_map

DO_NOT_USE = 1
NON_SCIENCE = 512
VARIANCES = ["VAR_POISSON", "VAR_RNOISE", "VAR_FLAT"]
DECADES = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]


class Exposure:
    """Science data, variance planes, weights and WCS of one exposure, in float64."""

    def __init__(self, path, weight_type):
        self.path = path
        with fits.open(path) as hdus:
            self.data = hdus["SCI"].data.astype(np.float64)
            self.variances = [hdus[name].data.astype(np.float64) for name in VARIANCES]
            dq = hdus["DQ"].data
            exptime = hdus[0].header["EFFEXPTM"]
            tree = hdus["ASDF"].data.tobytes()
        # the tree also refers to the FITS arrays, which asdf cannot resolve here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", asdf.exceptions.AsdfConversionWarning)
            with asdf.open(io.BytesIO(tree), lazy_load=False, memmap=False) as af:
                self.wcs = af.tree["meta"]["wcs"]
        good = (dq & (DO_NOT_USE | NON_SCIENCE)) == 0
        if weight_type == "ivm":
            var = self.variances[1]
            good &= np.isfinite(var) & (var > 0)
            self.weights = np.zeros(var.shape)
            self.weights[good] = 1.0 / var[good]
        else:
            self.weights = np.where(good, exptime, 0.0)


def output_frame(exposures, pixel_scale_ratio, margin=2):
    """North-up TAN WCS and shape enclosing all exposures.

    The pixel scale is the first exposure's at its center times
    ``pixel_scale_ratio``.
    """
    ny, nx = exposures[0].data.shape
    ra, dec = exposures[0].wcs.pixel_to_world_values((nx - 1) / 2, (ny - 1) / 2)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]

    # area of the central input pixel in arcsec, from a 1 arcsec grid
    xc, yc = (nx - 1) / 2, (ny - 1) / 2
    x, y = wcs.world_to_pixel_values(*exposures[0].wcs.pixel_to_world_values([xc, xc + 1, xc], [yc, yc, yc + 1]))
    scale = np.sqrt(abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))) * pixel_scale_ratio
    wcs.wcs.cdelt = [-scale / 3600, scale / 3600]

    # input pixel edges, sampled along the outline of each exposure
    xs, ys = [], []
    for exposure in exposures:
        ny, nx = exposure.data.shape
        side_x = np.arange(-0.5, nx, 16.0)
        side_y = np.arange(-0.5, ny, 16.0)
        px = np.concatenate([side_x, side_x, np.full(side_y.size, -0.5), np.full(side_y.size, nx - 0.5)])
        py = np.concatenate([np.full(side_x.size, -0.5), np.full(side_x.size, ny - 0.5), side_y, side_y])
        x, y = wcs.world_to_pixel_values(*exposure.wcs.pixel_to_world_values(px, py))
        xs.append(x)
        ys.append(y)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    wcs.wcs.crpix = [1.0 + margin - x.min(), 1.0 + margin - y.min()]
    shape = (int(np.ceil(y.max() - y.min())) + 2 * margin + 1, int(np.ceil(x.max() - x.min())) + 2 * margin + 1)
    wcs.pixel_shape = shape[::-1]
    return wcs, shape


def run_drizzle(exposures, pixmaps, out_shape, pixfrac, dtype):
    """Drizzle all exposures with tdriz into output arrays of ``dtype``; return the outputs and the time of each call by step."""
    out = {
        "sci": np.full(out_shape, np.nan, dtype=dtype),
        "wht": np.zeros(out_shape, dtype=dtype),
        "ctx": np.zeros(out_shape, dtype=np.int32),
        "var": [np.full(out_shape, np.nan, dtype=dtype) for _ in VARIANCES],
    }
    times = []
    for uniqid, (exposure, pixmap) in enumerate(zip(exposures, pixmaps), start=1):
        data = exposure.data.astype(dtype)
        weights = exposure.weights.astype(dtype)
        variances = [var.astype(dtype) for var in exposure.variances]
        t0 = time.perf_counter()
        cdrizzle.tdriz(
            data,
            weights,
            pixmap,
            out["sci"],
            out["wht"],
            out["ctx"],
            input2=variances,
            output2=out["var"],
            uniqid=uniqid,
            pixfrac=pixfrac,
            kernel="square",
            in_units="cps",
            expscale=1.0,
            wtscale=1.0,
            fillstr="INDEF",
        )
        times.append(time.perf_counter() - t0)
    return out, {f"drizzle tdriz, {np.dtype(dtype).name} outputs, science + 3 variance planes": times}


def run_spizzle(exposures, pixmaps, out_shape, pixfrac):
    """Co-add all exposures with spizzle; return the Coadd and the operator and add times of each exposure by step."""
    coadd = Coadd(out_shape, n_var=len(VARIANCES))
    operator_times = []
    add_times = []
    for exposure, pixmap in zip(exposures, pixmaps):
        t0 = time.perf_counter()
        W = drizzle_operator(pixmap, out_shape, pixfrac=pixfrac, weights=exposure.weights)
        t1 = time.perf_counter()
        coadd.add(W, exposure.data, exposure.variances)
        t2 = time.perf_counter()
        operator_times.append(t1 - t0)
        add_times.append(t2 - t1)
    return coadd, {
        "spizzle drizzle_operator": operator_times,
        "spizzle Coadd.add, science + 3 variance planes": add_times,
    }


def decade_counts(ours, reference, weighted):
    """Pixels with weight per decade of ``|ours - reference| / max(|reference|, median |reference|)``.

    Returns the counts of exact equality, of each decade from below 1e-16
    to above 1e-6, and of pixels finite in one array and not the other.
    """
    both = weighted & np.isfinite(ours) & np.isfinite(reference)
    scale = np.maximum(np.abs(reference[both]), np.median(np.abs(reference[both])))
    rel = np.abs(ours[both] - reference[both]) / scale
    exact = int(np.count_nonzero(rel == 0.0))
    edges = np.concatenate([[0.0], DECADES, [np.inf]])
    counts = np.histogram(rel[rel > 0.0], bins=edges)[0]
    mismatch = int(np.count_nonzero(weighted & (np.isfinite(ours) != np.isfinite(reference))))
    return [exact, *counts.tolist(), mismatch], float(rel.max()) if rel.size else 0.0


def comparison_table(coadd, reference, weighted):
    """Markdown table of decade_counts for each output plane against one drizzle run."""
    heads = ["exact", "<1e-16"] + [f"{lo:.0e}" for lo in DECADES[:-1]] + [">=1e-6", "NaN differs", "max"]
    lines = ["| plane | " + " | ".join(heads) + " |", "| --- |" + " ---: |" * len(heads)]
    planes = [("sci", coadd.out_img, reference["sci"]), ("wht", coadd.out_wht, reference["wht"])]
    planes += [(name.lower(), ours, ref) for name, ours, ref in zip(VARIANCES, coadd.out_var, reference["var"])]
    for name, ours, ref in planes:
        counts, largest = decade_counts(ours, ref.astype(np.float64), weighted)
        lines.append(f"| {name} | " + " | ".join(str(c) for c in counts) + f" | {largest:.1e} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("files", nargs="+", help="cal or crf files of one detector and filter")
    parser.add_argument("--pixfrac", type=float, default=1.0)
    parser.add_argument("--pixel-scale-ratio", type=float, default=1.0, help="output over input pixel scale")
    parser.add_argument("--weight-type", choices=["ivm", "exptime"], default="ivm")
    parser.add_argument("--repeat", type=int, default=3, help="passes of each package over all exposures")
    args = parser.parse_args()

    t0 = time.perf_counter()
    exposures = [Exposure(path, args.weight_type) for path in args.files]
    t_read = time.perf_counter() - t0
    wcs_out, out_shape = output_frame(exposures, args.pixel_scale_ratio)
    print(f"{len(exposures)} exposures, read in {t_read:.2f} s; output {out_shape[0]}x{out_shape[1]}, "
          f"{3600 * abs(wcs_out.wcs.cdelt[0]):.4f} arcsec per pixel, pixfrac {args.pixfrac}, weights {args.weight_type}")

    pixmap_times = []
    pixmaps = []
    for exposure in exposures:
        t0 = time.perf_counter()
        pixmaps.append(calc_pixmap(exposure.wcs, wcs_out, shape=exposure.data.shape))
        pixmap_times.append(time.perf_counter() - t0)

    spline_times = []
    spline_error = 0.0
    for exposure, pixmap in zip(exposures, pixmaps):
        t0 = time.perf_counter()
        interpolated = pixel_map(exposure.wcs, wcs_out, shape=exposure.data.shape)
        spline_times.append(time.perf_counter() - t0)
        spline_error = max(spline_error, float(np.nanmax(np.abs(interpolated - pixmap))))
    print(f"spizzle pixel_map differs from calc_pixmap by at most {spline_error:.1e} pixel")

    # compile and load the numba kernels before timing
    y, x = np.indices((4, 4), dtype=np.float64)
    Coadd((4, 4), n_var=1).add(drizzle_operator(np.dstack([x, y]), (4, 4)), np.ones((4, 4)), [np.ones((4, 4))])

    runs = {
        "spizzle": lambda: run_spizzle(exposures, pixmaps, out_shape, args.pixfrac),
        "float64": lambda: run_drizzle(exposures, pixmaps, out_shape, args.pixfrac, np.float64),
        "float32": lambda: run_drizzle(exposures, pixmaps, out_shape, args.pixfrac, np.float32),
    }
    # timed passes keep only one package's outputs in memory, so that a large
    # output frame does not run the machine short of memory while timing
    times = {
        "pixel map (gwcs, calc_pixmap)": np.array(pixmap_times),
        "pixel map (spizzle pixel_map, bicubic every 16 pixels)": np.array(spline_times),
    }
    order = list(runs)
    for _ in range(args.repeat):
        for name in order:
            output, step_times = runs[name]()
            del output
            for step, t in step_times.items():
                times[step] = np.minimum(times[step], t) if step in times else np.array(t)
        order.reverse()
    coadd, drizzle64, drizzle32 = (runs[name]()[0] for name in ["spizzle", "float64", "float32"])

    weighted = drizzle64["wht"] > 0
    print(f"\n{np.count_nonzero(weighted)} output pixels with weight; spizzle has weight in "
          f"{np.count_nonzero(coadd.out_wht > 0)}")
    for name, reference in [("float64", drizzle64), ("float32", drizzle32)]:
        print(f"\nspizzle against drizzle with {name} outputs, pixels per decade of relative difference:\n")
        print(comparison_table(coadd, reference, weighted))
        same = np.array_equal(coadd.out_ctx[0], reference["ctx"])
        print(f"\ncontext identical: {same}")

    flux_in = sum(np.sum(e.data[e.weights > 0] * e.weights[e.weights > 0]) for e in exposures)
    print(f"\nflux sum(sci * wht) / sum(data * weights): spizzle {np.nansum(coadd.out_img * coadd.out_wht) / flux_in:.15f}, "
          f"drizzle float64 {np.nansum(drizzle64['sci'] * drizzle64['wht']) / flux_in:.15f}, "
          f"drizzle float32 {np.nansum(drizzle32['sci'].astype(np.float64) * drizzle32['wht']) / flux_in:.15f}")

    print(f"\n| step, minimum of {args.repeat} passes | mean per exposure | total |")
    print("| --- | ---: | ---: |")
    for step, t in times.items():
        print(f"| {step} | {np.mean(t):.3f} s | {np.sum(t):.2f} s |")


if __name__ == "__main__":
    main()
