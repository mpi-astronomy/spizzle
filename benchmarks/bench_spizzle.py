"""
Timing benchmarks for spizzle against drizzle's C square kernel.

Run from the repository root, with drizzle installed (it must accept float64
output arrays, see the README)::

    python benchmarks/bench_spizzle.py

The input is a detector-sized image with a rotated and slightly distorted
pixel map, as in drizzle's ``benchmarks/bench_cdrizzle.py``, drizzled onto
output pixels ``--oversample`` times smaller, in an output frame that
encloses the whole input.  The cases are:

- ``tdriz_<dtype>_planes<k>``: one ``cdrizzle.tdriz`` call with the science
  data and ``k - 1`` variance planes, with the float32 output arrays of
  released drizzle and with float64 ones;
- ``spizzle_operator_threads<n>``: ``drizzle_operator`` on ``n`` numba
  threads;
- ``spizzle_corners``, ``spizzle_jacobian``, ``spizzle_triplets`` and
  ``spizzle_build``: the steps of ``drizzle_operator``, on all threads;
- ``spizzle_add_planes<k>``: ``Coadd.add`` of the science data and ``k - 1``
  variance planes, with the operator built beforehand.

Output arrays are written before the timed runs, so that page faults on
first touch are not timed; drizzle and ``Coadd`` both reuse their outputs
from one exposure to the next.  Each case is timed ``--repeat`` times after
one untimed run, and the summary uses the minimum time, which is the least
sensitive to other load on the machine.  ``--rss`` also runs each package
once with three variance planes in a fresh process and reports its peak
resident memory.  ``--save`` and ``--compare`` keep JSON baselines.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import warnings

import numba
import numpy as np
import scipy

import drizzle
from drizzle import cdrizzle

import spizzle
from spizzle import Coadd, build_operator, corners, drizzle_operator, jacobian, overlap_triplets

DTYPES = ["float32", "float64"]
RSS_CASES = ["tdriz_float32_planes4", "tdriz_float64_planes4", "spizzle_planes4"]


class Inputs:
    """Science data, variance planes and pixel map of one exposure, and the output shape."""

    def __init__(self, size, oversample, nvar):
        rng = np.random.default_rng(12345)
        self.data = rng.random((size, size)) + 1.0
        self.variances = [rng.random((size, size)) + 1.0 for _ in range(nvar)]
        self.weights = np.ones((size, size))
        y, x = np.indices((size, size), dtype=np.float64)
        xc = x - size / 2
        yc = y - size / 2
        xo = oversample * (0.96 * xc - 0.28 * yc + 1.0e-5 * xc * xc)
        yo = oversample * (0.28 * xc + 0.96 * yc + 1.0e-5 * xc * yc)
        # a margin of 4 output pixels, so that drizzle visits every input pixel too
        self.pixmap = np.dstack([xo - xo.min() + 4.0, yo - yo.min() + 4.0])
        self.out_shape = (int(np.ceil(np.ptp(yo))) + 8, int(np.ceil(np.ptp(xo))) + 8)


def tdriz_case(inputs, pixfrac, dtype, planes):
    def setup():
        data = inputs.data.astype(dtype)
        weights = inputs.weights.astype(dtype)
        out = np.full(inputs.out_shape, 0.0, dtype=dtype)
        counts = np.full(inputs.out_shape, 0.0, dtype=dtype)
        context = np.full(inputs.out_shape, 0, dtype=np.int32)
        kwargs = {}
        if planes > 1:
            kwargs["input2"] = [var.astype(dtype) for var in inputs.variances[:planes - 1]]
            kwargs["output2"] = [np.full(inputs.out_shape, 0.0, dtype=dtype) for _ in range(planes - 1)]

        def run():
            t0 = time.perf_counter()
            cdrizzle.tdriz(
                data,
                weights,
                inputs.pixmap,
                out,
                counts,
                context,
                uniqid=1,
                pixfrac=pixfrac,
                kernel="square",
                in_units="cps",
                expscale=1.0,
                wtscale=1.0,
                fillstr="INDEF",
                **kwargs,
            )
            return time.perf_counter() - t0

        return run

    return setup


def operator_case(inputs, pixfrac, threads):
    def setup():
        def run():
            numba.set_num_threads(threads)
            t0 = time.perf_counter()
            drizzle_operator(inputs.pixmap, inputs.out_shape, pixfrac=pixfrac, weights=inputs.weights)
            return time.perf_counter() - t0

        return run

    return setup


def step_case(inputs, pixfrac, step):
    def setup():
        c = corners(inputs.pixmap, pixfrac)
        jaco = jacobian(c)
        triplets = overlap_triplets(c, inputs.out_shape)
        call = {
            "corners": lambda: corners(inputs.pixmap, pixfrac),
            "jacobian": lambda: jacobian(c),
            "triplets": lambda: overlap_triplets(c, inputs.out_shape),
            "build": lambda: build_operator(triplets, jaco, inputs.out_shape, inputs.weights),
        }[step]

        def run():
            t0 = time.perf_counter()
            call()
            return time.perf_counter() - t0

        return run

    return setup


def add_case(inputs, pixfrac, planes):
    def setup():
        W = drizzle_operator(inputs.pixmap, inputs.out_shape, pixfrac=pixfrac, weights=inputs.weights)
        coadd = Coadd(inputs.out_shape, n_var=planes - 1)
        variances = inputs.variances[:planes - 1]

        def run():
            t0 = time.perf_counter()
            coadd.add(W, inputs.data, variances)
            return time.perf_counter() - t0

        return run

    return setup


def make_cases(inputs, pixfrac, planes, threads):
    cases = {}
    for dtype in DTYPES:
        for k in planes:
            cases[f"tdriz_{dtype}_planes{k}"] = tdriz_case(inputs, pixfrac, dtype, k)
    for n in threads:
        cases[f"spizzle_operator_threads{n}"] = operator_case(inputs, pixfrac, n)
    for step in ["corners", "jacobian", "triplets", "build"]:
        cases[f"spizzle_{step}"] = step_case(inputs, pixfrac, step)
    for k in planes:
        cases[f"spizzle_add_planes{k}"] = add_case(inputs, pixfrac, k)
    return cases


def time_case(setup, repeat):
    """Time a case ``repeat`` times after one untimed run."""
    run = setup()
    run()
    times = [run() for _ in range(repeat)]
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    return {"min": min(times), "median": statistics.median(times)}


def peak_rss():
    """Peak resident memory of this process in bytes."""
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def measure_rss(name, args):
    """Run one RSS case once and return the peak resident memory before and after, in bytes."""
    inputs = Inputs(args.size, args.oversample, nvar=3)
    before = peak_rss()
    if name == "spizzle_planes4":
        W = drizzle_operator(inputs.pixmap, inputs.out_shape, pixfrac=args.pixfrac, weights=inputs.weights)
        Coadd(inputs.out_shape, n_var=3).add(W, inputs.data, inputs.variances)
    else:
        _, dtype, planes = name.split("_")
        tdriz_case(inputs, args.pixfrac, dtype, int(planes.removeprefix("planes")))()()
    return {"before": before, "peak": peak_rss()}


def rss_in_subprocess(name, args):
    command = [
        sys.executable, __file__, "--rss-child", name,
        "--size", str(args.size), "--oversample", str(args.oversample), "--pixfrac", str(args.pixfrac),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def summary(results, planes, threads):
    """Markdown table of the minimum times for each number of planes, with empty cells for cases not run."""

    def best(name):
        return results[name]["min"] if name in results else None

    operator = best(f"spizzle_operator_threads{max(threads)}")
    one_plane = best("tdriz_float32_planes1")
    lines = [
        "| planes | drizzle float32, one call | drizzle float64, one call | drizzle float32, one call per plane "
        f"| spizzle, operator on {max(threads)} threads + add | drizzle float32 one call / spizzle |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in planes:
        one_call = best(f"tdriz_float32_planes{k}")
        add = best(f"spizzle_add_planes{k}")
        total = None if operator is None or add is None else operator + add
        row = [
            one_call,
            best(f"tdriz_float64_planes{k}"),
            None if one_plane is None else k * one_plane,
            total,
        ]
        cells = ["" if seconds is None else f"{seconds:.3f} s" for seconds in row]
        cells.append("" if one_call is None or total is None else f"{one_call / total:.2f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--size", type=int, default=2048, help="input image size in pixels")
    parser.add_argument("--oversample", type=float, default=2.0, help="input pixel size in output pixels")
    parser.add_argument("--pixfrac", type=float, default=0.8)
    parser.add_argument("--planes", type=int, nargs="+", default=[1, 4, 8, 16], help="science plus variance planes")
    parser.add_argument("--threads", type=int, nargs="+", help="numba thread counts (default 1, 4, 8 and all)")
    parser.add_argument("--repeat", type=int, default=5, help="timings per case")
    parser.add_argument("-k", dest="select", help="only run cases containing this string")
    parser.add_argument("--rss", action="store_true", help="also measure peak memory with three variance planes")
    parser.add_argument("--save", metavar="FILE", help="write results to a JSON file")
    parser.add_argument("--compare", metavar="FILE", help="compare with saved JSON results")
    parser.add_argument("--rss-child", metavar="CASE", choices=RSS_CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.rss_child:
        print(json.dumps(measure_rss(args.rss_child, args)))
        return

    max_threads = numba.config.NUMBA_NUM_THREADS
    threads = args.threads or sorted({n for n in [1, 4, 8] if n < max_threads} | {max_threads})
    if max(threads) > max_threads:
        parser.error(f"numba has {max_threads} threads; set NUMBA_NUM_THREADS for more")

    baseline = None
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        for key in ["size", "oversample", "pixfrac"]:
            if baseline["meta"][key] != getattr(args, key):
                parser.error(f"baseline was run with --{key} {baseline['meta'][key]}")

    inputs = Inputs(args.size, args.oversample, nvar=max(args.planes) - 1)
    W = drizzle_operator(inputs.pixmap, inputs.out_shape, pixfrac=args.pixfrac, weights=inputs.weights).matrix
    nnz = W.nnz
    W_bytes = W.data.nbytes + W.indices.nbytes + W.indptr.nbytes
    del W
    print(
        f"input {args.size}x{args.size}, output {inputs.out_shape[0]}x{inputs.out_shape[1]}, pixfrac {args.pixfrac}; "
        f"operator {nnz:.3g} non-zeros ({nnz / inputs.data.size:.2f} per input pixel), {W_bytes / 2**30:.2f} GiB"
    )

    cases = make_cases(inputs, args.pixfrac, args.planes, threads)
    if args.select:
        cases = {name: case for name, case in cases.items() if args.select in name}

    header = f"{'case':28s} {'min (s)':>9s} {'median (s)':>11s}"
    if baseline:
        header += f" {'base min':>9s} {'ratio':>7s}"
    print(header)

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, setup in cases.items():
            this = time_case(setup, args.repeat)
            results[name] = this
            line = f"{name:28s} {this['min']:9.4f} {this['median']:11.4f}"
            if baseline and name in baseline["results"]:
                base = baseline["results"][name]["min"]
                line += f" {base:9.4f} {this['min'] / base:7.3f}"
            print(line, flush=True)

    if not args.select:
        print()
        print(summary(results, args.planes, threads))

    rss = {}
    if args.rss:
        print()
        print(f"{'case (one run, fresh process)':32s} {'peak (GiB)':>10s} {'above inputs':>13s}")
        for name in RSS_CASES:
            rss[name] = rss_in_subprocess(name, args)
            peak = rss[name]["peak"] / 2**30
            above = (rss[name]["peak"] - rss[name]["before"]) / 2**30
            print(f"{name:32s} {peak:10.2f} {above:13.2f}", flush=True)

    if args.save:
        meta = {
            "size": args.size,
            "oversample": args.oversample,
            "pixfrac": args.pixfrac,
            "repeat": args.repeat,
            "out_shape": inputs.out_shape,
            "nnz": nnz,
            "numba_threads": max_threads,
            "spizzle": spizzle.__version__,
            "drizzle": drizzle.__version__,
            "numba": numba.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        }
        with open(args.save, "w") as f:
            json.dump({"meta": meta, "results": results, "rss": rss}, f, indent=2)
        print(f"Wrote {args.save}")


if __name__ == "__main__":
    main()
