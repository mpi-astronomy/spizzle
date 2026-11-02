"""The speed table of the README: drizzle and spizzle on one JWST exposure, step by step.

Run from the repository root::

    python benchmarks/speed_table.py DATA/jw*_nrcalong_cal.fits [--jwst]

The first file is the exposure; all of them define the output frame, a
native-scale mosaic enclosing them (as in ``compare_jwst.py``).  Each step
is the best of ``--repeat`` passes:

- the pixel map from the WCS: drizzle's ``calc_pixmap`` (every pixel)
  against spizzle's ``pixel_map`` (splines through every 16th pixel);
- drizzling the science data with its three variance planes: one
  ``tdriz`` call with float32 outputs, as in released drizzle, against
  ``drizzle_operator`` and ``Coadd.add``;
- blotting a median and its error back onto the exposure, given the
  pixel map: two drizzle ``blot_image`` calls against two ``blot`` calls.

With ``--jwst`` (needs the ``jwst`` extra), also jwst's ``ResampleImage`` of
all the files against spizzle with ``StcalVariance``, from
``compare_stcal.py``.
"""
import argparse
import sys
import time
import warnings

import numba
import numpy as np
from drizzle import cdrizzle
from drizzle.resample import blot_image
from drizzle.utils import calc_pixmap

from spizzle import Coadd, blot, drizzle_operator, pixel_map

sys.path.insert(0, "benchmarks")
from compare_jwst import Exposure, output_frame  # noqa: E402


def best(f, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("files", nargs="+", help="JWST cal files; the first is timed")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--jwst", action="store_true", help="also time jwst's ResampleImage of all the files")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = [Exposure(f, "ivm") for f in args.files]
        wcs_out, out_shape = output_frame(frames, 1.0)
        e = frames[0]
        shape = e.data.shape
        pixmap = pixel_map(e.wcs, wcs_out, shape=shape)

        rows = []
        t_drizzle = best(lambda: calc_pixmap(e.wcs, wcs_out, shape=shape), args.repeat)
        t_spizzle = best(lambda: pixel_map(e.wcs, wcs_out, shape=shape), args.repeat)
        rows.append(("pixel map from the WCS", t_drizzle, t_spizzle))

        data = e.data.astype(np.float32)
        weights = e.weights.astype(np.float32)
        variances = [v.astype(np.float32) for v in e.variances]

        def run_drizzle():
            cdrizzle.tdriz(data, weights, pixmap, np.full(out_shape, np.nan, dtype=np.float32),
                           np.zeros(out_shape, dtype=np.float32), np.zeros(out_shape, dtype=np.int32),
                           input2=variances, output2=[np.full(out_shape, np.nan, dtype=np.float32) for _ in variances],
                           uniqid=1, pixfrac=1.0, kernel="square", in_units="cps", expscale=1.0, wtscale=1.0,
                           fillstr=str(np.nan))

        def run_spizzle():
            coadd = Coadd(out_shape, n_var=3)
            coadd.add(drizzle_operator(pixmap, out_shape, weights=e.weights), e.data, e.variances)

        rows.append(("drizzle the science data and 3 variance planes", best(run_drizzle, args.repeat),
                     best(run_spizzle, args.repeat)))

        y, x = np.indices(out_shape)
        median = (np.sin(x / 37.0) * np.cos(y / 53.0) + 2.0).astype(np.float32)
        error = (0.1 + 0.01 * np.cos(x / 29.0)).astype(np.float32)
        blot_map = np.where(np.isnan(pixmap), -1.0, pixmap)  # as stcal's gwcs_blot passes it

        def run_blot_image():
            for image in (median, error):
                blot_image(image, blot_map, out_img=np.full(shape, np.nan, dtype=np.float32), fillval=np.nan,
                           interp="linear")

        def run_blot():
            for image in (median, error):
                blot(image, pixmap, out=np.empty(shape, dtype=np.float32))

        rows.append(("blot a median and its error, given the pixel map", best(run_blot_image, args.repeat),
                     best(run_blot, args.repeat)))

        if args.jwst:
            from compare_stcal import run_jwst, run_spizzle as run_spizzle_stcal
            output, resamp, library, t_jwst = run_jwst(args.files)
            _, _, t_stcal = run_spizzle_stcal(library, resamp, output.data.shape, 16)
            rows.append((f"jwst ResampleImage of {len(args.files)} exposures (spizzle: StcalVariance)", t_jwst, t_stcal))

    print(f"\n{numba.get_num_threads()} threads; one {shape[0]}x{shape[1]} exposure onto a {out_shape[0]}x{out_shape[1]} "
          f"mosaic, best of {args.repeat}\n")
    print("| step | drizzle | spizzle | faster |")
    print("| --- | ---: | ---: | ---: |")
    for name, a, b in rows:
        print(f"| {name} | {a:.3g} s | {b:.3g} s | {a / b:.0f}x |")


if __name__ == "__main__":
    main()
