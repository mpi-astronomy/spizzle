"""Resample JWST ``cal`` files with jwst's ResampleImage and with spizzle, and compare every output plane.

Run from the repository root, with the ``jwst`` extra installed
(``pip install -e ".[jwst]"``, jwst 2.0 or later)::

    python benchmarks/compare_stcal.py DATA/jw*_nrcalong_cal.fits

jwst resamples the exposures onto its default output frame (square kernel,
pixfrac 1, ivm weights, the pixel map of every pixel), with the variance
planes and ERR computed from them.  spizzle then drizzles the same
exposures onto the same frame, with stcal's weights (``build_driz_weight``)
and input ranges (``resample_range``): `Coadd` for the science image,
weights and context, and `StcalVariance` for the variance planes and ERR
as stcal computes them.  For each plane, the table gives the relative
difference over the output pixels with weight, and the pixels where only
one of them is NaN.  stcal keeps its outputs in float32, spizzle in
float64.
"""
import argparse
import time
import warnings

import numpy as np

from spizzle import Coadd, StcalVariance, drizzle_operator, pixel_map

VARIANCES = ["var_rnoise", "var_flat", "var_poisson"]


def run_jwst(files):
    """jwst's ResampleImage of the files; the output model, the ResampleImage, and the time it took."""
    from jwst import datamodels
    from jwst.datamodels import ModelLibrary
    from jwst.resample.resample import ResampleImage
    from jwst.resample.resample_step import GOOD_BITS

    library = ModelLibrary([datamodels.open(f) for f in files], on_disk=False)
    t0 = time.perf_counter()
    resamp = ResampleImage(library, weight_type="ivm", good_bits=GOOD_BITS, blendheaders=False, enable_ctx=True,
                           enable_var=True, report_var=True, compute_err="from_var")
    output = resamp.resample_many_to_one()
    return output, resamp, library, time.perf_counter() - t0


def run_spizzle(library, resamp, out_shape, stepsize):
    """spizzle's Coadd and StcalVariance of the same exposures, with stcal's weights and input ranges, and the time."""
    from jwst.resample.resample import input_jwst_model_to_dict
    from jwst.resample.resample_step import GOOD_BITS
    from stcal.resample.utils import build_driz_weight, resample_range

    coadd = Coadd(out_shape, n_var=3)
    sv = StcalVariance(out_shape, weight_type="ivm", pixel_scale_ratio=resamp.pixel_scale_ratio)
    elapsed = 0.0
    with library:
        for model in library:
            m = input_jwst_model_to_dict(model, "ivm", True, "from_var")
            t0 = time.perf_counter()
            weights = build_driz_weight(m, weight_type="ivm", good_bits=GOOD_BITS, flag_name_map=resamp.dq_flag_name_map)
            data = m["data"]
            if not m["subtracted"] and m["level"] is not None:
                data = data - m["level"]
            pixmap = pixel_map(m["wcs"], resamp.output_wcs, shape=data.shape, stepsize=stepsize)
            xmin, xmax, ymin, ymax = resample_range(data.shape, m["wcs"].bounding_box)
            op = drizzle_operator(pixmap, out_shape, weights=weights,
                                  region=(slice(ymin, ymax + 1), slice(xmin, xmax + 1)))
            variances = [m[name] for name in VARIANCES]
            coadd.add(op, data, variances)
            sv.add(op, variances)
            elapsed += time.perf_counter() - t0
            library.shelve(model, modify=False)
    return coadd, sv, elapsed


def compare(name, ours, theirs, weighted):
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    both = weighted & np.isfinite(a) & np.isfinite(b) & (b != 0)
    rel = np.abs(a[both] / b[both] - 1.0)
    one_nan = int(np.sum(weighted & (np.isfinite(a) != np.isfinite(b))))
    print(f"| {name} | {np.median(rel):.1e} | {np.percentile(rel, 99.9):.1e} | {rel.max():.1e} | {one_nan} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("files", nargs="+", help="JWST cal files")
    parser.add_argument("--stepsize", type=int, default=1, help="pixel_map stepsize for spizzle (jwst evaluates every pixel)")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output, resamp, library, t_jwst = run_jwst(args.files)
        out_shape = output.data.shape
        coadd, sv, t_spizzle = run_spizzle(library, resamp, out_shape, args.stepsize)

    print(f"\n{len(args.files)} exposures onto jwst's {out_shape[0]}x{out_shape[1]} frame; jwst ResampleImage {t_jwst:.1f} s, "
          f"spizzle {t_spizzle:.1f} s (pixel_map stepsize {args.stepsize}, operator, Coadd, StcalVariance)\n")
    weighted = output.wht > 0
    print(f"relative difference over the {weighted.sum()} output pixels with weight:\n")
    print("| plane | median | 99.9% | max | NaN in one only |")
    print("| --- | ---: | ---: | ---: | ---: |")
    compare("SCI", coadd.out_img, output.data, weighted)
    compare("WHT", coadd.out_wht, output.wht, weighted)
    for name, plane in zip(VARIANCES, sv.out_var):
        compare(name.upper(), plane, getattr(output, name), weighted)
    compare("ERR", sv.out_err, output.err, weighted)
    con_same = np.array_equal(coadd.out_ctx, output.con)
    print(f"\nCON equal: {con_same}")


if __name__ == "__main__":
    main()
