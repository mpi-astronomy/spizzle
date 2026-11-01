# Plan: faster, float64, parallel drizzle via an explicit sparse operator

**Status:** milestone steps 1-2 done (`fc5db6d`); step 3 done: baseline in
§8.1, after the speed-up of the operator build and apply in §8.2; step 4
(A vs B, the B.1 PR) open
**Context date:** 2026-11-01
**Reference code:** `spacetelescope/drizzle` `main` @ `59ff6a1` (2026-10-28)

## 1. Problem statement

`jwst.resample.ResampleStep` is backed by the `drizzle` package. Its C kernel
(`src/cdrizzlebox.c`) is single-threaded, stores and re-normalises output
pixels in float32 on every contribution, and recomputes the input→output
pixel-overlap geometry for every data plane pushed through it (science, each
variance plane, every blot/outlier-detection iteration, any noise realisations).

Drizzle is a linear map `y = W x`, where `W` is sparse (~4–9 non-zeros per
input pixel) and depends only on the WCS pair, `pixfrac`, output grid and input
weights — not on the data. Computing `W` once per exposure and applying it with
`scipy.sparse` in float64 removes the redundant geometry work, fixes the
precision problem, makes the geometry stage embarrassingly parallel, and gives
an explicit operator from which output noise covariance (`W Σ Wᵀ`) can be
derived.

## 2. Goals

1. Numerically identical (to float64 round-off) science/weight output to
   `drizzle` square kernel for the same inputs.
2. Float64 accumulation; one normalisation at the end.
3. Wall-clock speed-up when ≥2 planes share geometry; multithreaded geometry
   build.
4. Expose `W` (or its per-exposure pieces) so covariance / noise-propagation
   can be computed downstream.

## 3. Non-goals (for now)

- Replacing drizzle's resampling *model*. Square kernel + `pixfrac` only.
  `point`/`gaussian`/`lanczos`/`turbo` kernels out of scope.
- Changing the `jwst` pipeline interface. First target is a standalone callable
  that a user script can invoke on `cal.fits` + gwcs.
- IMCOM-style PSF-matched coaddition (see pyimcom for that).

## 4. What the current C code does (read before changing anything)

File: `src/cdrizzlebox.c`

- `do_kernel_square_var(p)` — the default path. Per input pixel `(i, j)`:
  1. `interpolate_four_points` (`src/cdrizzlemap.c`): bilinear interpolation of
     the 4 corners of the shrunken pixel from the precomputed `pixmap` array
     (9 strided `PyArray_GETPTR2` reads).
  2. Jacobian from the 4 output-plane corners; weight `w = wt(i,j)·weight_scale/jaco`.
  3. Bounding box of the quadrilateral in output pixels; for each candidate
     output pixel call `boxer(ii, jj, xout, yout)`.
  4. `boxer` = sum of 4 `sgarea` calls (area under a segment clipped to the unit
     square). Pure double arithmetic, branchy.
  5. `update_data_var`: running weighted mean
     `v = (old·vc + dow·d)/(vc+dow)` for science, and the squared-weight
     analogue for every `data2` (variance) plane. One division per plane per
     hit; result stored as float32. `dow` is cast to float before use.
- Build: `setup.py` adds only `-O3`. No OpenMP, no threads.
- Python entry: `drizzle.resample.Drizzle.add_image` → `cdrizzle.tdriz`, called
  once per input image; `data2` arrays (variance planes) are co-added in the same
  pass with squared weights.

Key facts for the rewrite:
- All geometry is already double precision. The float32 loss is purely from the
  running-mean update in `update_data_var`.
- `boxer`/`sgarea` is the only piece that genuinely resists numpy
  vectorisation (branches on clipping). Everything else is array arithmetic.

## 5. Proposed architecture

```
corners(gwcs_in, gwcs_out, pixfrac)        # numpy: (ny, nx, 4, 2) float64
    │
overlap_triplets(corners, out_shape)       # compiled: numba first, C later
    │   returns (in_idx, out_idx, area)  int64/int64/float64
build_operator(triplets, weights, jaco)    # scipy.sparse CSR, shape (n_out, n_in)
    │
apply(W, planes)                           # W @ sci, W @ ones, (W∘W) @ var, ...
    │
accumulate across exposures in float64; normalise once at end
```

### 5.1 `corners`
- Evaluate the input→output transform directly on a corner grid
  (`x ± pixfrac/2, y ± pixfrac/2`) via gwcs, **or** bilinearly interpolate the
  existing `pixmap` exactly as `interpolate_four_points` does. Do the latter
  first — it is what makes the equivalence test in §7 exact.
- Jacobian: `0.5*((x1-x3)(y0-y2) - (x0-x2)(y1-y3))`, same corner ordering as
  the C code (keep sign convention; areas and `jaco` cancel).

### 5.2 `overlap_triplets`
- Port `boxer` + `sgarea` verbatim into `@numba.njit(parallel=True)`.
  Loop over input pixels (`prange` over rows), bounding box → candidate output
  pixels → area. Append non-zero areas to per-row buffers, concatenate.
- Edge cases to preserve: `dover != 0` test (areas may be negative when
  `jaco < 0`; sign cancels in `dow = dover*w`), output-bounds clipping
  (`MAX(.., 0)`, `MIN(.., osize-1)` with `nintd` rounding), pixels with
  weight 0 or NaN data skipped.
- Expected size: 2048² input at 2× oversampling → ~3.5e7 non-zeros,
  ~0.5 GB CSR. Process per exposure and discard.

### 5.3 `build_operator`
- Entries: `area · weight(i,j) · weight_scale / jaco(i,j)`.
- `scipy.sparse.coo_matrix((vals, (out_idx, in_idx)), shape=(n_out, n_in)).tocsr()`
  (duplicate summing in `tocsr` is what replaces the scatter-add).
- Keep `W_unweighted` (areas only) separately if you want to re-weight without
  rebuilding geometry.

### 5.4 `apply` and accumulation
Per exposure *k*, with `d = sci · iscale`, `v = var · iscale²`:
```
num   += W_k @ d
wsum  += W_k @ 1
vnum  += (W_k.multiply(W_k)) @ v        # one per variance plane
ctx    |= (W_k @ 1 > 0) << k            # context
```
Final: `sci_out = num/wsum`, `var_out = vnum/wsum²` where `wsum>0`, else fill.
This reproduces the C running-mean result exactly in exact arithmetic.

Blot (output→input) for outlier detection: same corner machinery with the
transforms swapped, or interpolation as in `cdrizzleblot.c` — decide later;
not needed for the first benchmark.

### 5.5 Covariance (stretch)
`Cov_out = W Σ_in Wᵀ` with `Σ_in` diagonal → `(W.multiply(W @ diag)) @ Wᵀ`
restricted to a band. Enough to quantify pixel–pixel correlation vs. the usual
Casertano correction factor.

## 6. Repository strategy

Two options; pick after the numba prototype works.

**A. Separate package** (recommended for the prototype)
- `pip install`-able, depends on numpy/scipy/numba/gwcs/astropy only.
- Imports `drizzle` solely in the test suite for equivalence checks.
- Fast to iterate; no CI/build-system friction.

**B. Branch of a `drizzle` fork**
- Natural home for two upstreamable pieces once proven:
  1. float64 accumulation + deferred normalisation in `update_data_var`
     (change output array dtype, remove per-hit division, normalise in Python
     after the last `add_image`). Small, self-contained, likely acceptable
     upstream.
  2. A new C entry point `boxer_grid(corners, out_shape) -> triplets`
     exposing the geometry without the scatter, so the sparse path can use the
     reference implementation instead of a port.
- Keep the existing API working; add the new path alongside.

Start with A; carry (B.1) as an independent PR regardless.

## 7. Validation

1. **Bit-for-bit geometry:** `overlap_triplets` vs. a tiny C shim around
   `boxer` on random quadrilaterals (or vs. `drizzle` run with a single
   input pixel non-zero). Tolerance 1e-12 absolute on areas.
2. **End-to-end equivalence:** synthetic image + `drizzle.resample.Drizzle`
   (square kernel, several `pixfrac`, rotation, scale ≠ 1, partial overlap,
   NaN/zero-weight pixels) vs. new path. Expect agreement at float32 round-off
   of the *reference*, not better; differences should shrink as the reference
   is run with fewer contributions per output pixel.
3. **Flux conservation:** Σ(sci_out·wsum) vs Σ(sci_in·wt) to 1e-10.
4. **Real data:** one NIRCam LW exposure set through both paths; compare
   science, weight and variance planes; histogram of relative differences.
5. **Variance plane:** confirm `vnum/wsum²` matches `drizzle`'s `data2`
   output.

## 8. Benchmarks to record

- Per exposure, 2048², 2× oversampling, `pixfrac=0.8`:
  - `drizzle` C: time for 1 plane; time for 4 planes (sci + 3 var).
  - New: geometry build (1, 4, 8 threads); apply per plane; peak RSS.
- Scaling with number of planes (1, 4, 8, 16) — the win should be ~linear.
- Note: the `ResampleStep` wall clock is often dominated by gwcs evaluation
  and outlier-detection round trips, not the kernel. Measure those
  separately so the comparison is honest.

### 8.1 Baseline, 2026-11-01 (spizzle `fc5db6d`)

`python benchmarks/bench_spizzle.py --rss`: 2048² input with a rotated,
slightly distorted pixel map, 2× oversampling, `pixfrac=0.8`, output
5085² enclosing the input. `W` has 3.16e7 non-zeros (7.53 per input pixel),
0.66 GiB. Minimum of 5 runs after one untimed run; output arrays written
beforehand in both packages. Apple M3 Pro (5 performance + 6 efficiency
cores, 18 GB), Python 3.14, numba 0.68.1, scipy 1.18.1, drizzle `f37a056`
(jdavies-st float64 branch, which takes float32 or float64 outputs).

Planes are the science data plus variance planes. "One call" passes the
variance planes to `tdriz` as `input2`, as jwst does; "one call per plane"
is k × the 1-plane time, for planes drizzled in separate passes.

| planes | drizzle f32, one call | drizzle f64, one call | drizzle f32, one call per plane | spizzle, `drizzle_operator` + `Coadd.add` | drizzle f32 one call / spizzle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.567 s | 0.600 s | 0.567 s | 1.142 s | 0.50 |
| 4 | 0.720 s | 0.765 s | 2.267 s | 1.501 s | 0.48 |
| 8 | 1.239 s | 0.950 s | 4.533 s | 1.789 s | 0.69 |
| 16 | 2.439 s | 2.321 s | 9.067 s | 3.866 s | 0.63 |

spizzle steps:

| step | time |
| --- | ---: |
| `drizzle_operator`, numba threads 1 / 4 / 8 / 11 | 1.48 / 0.99 / 0.96 / 0.93 s |
| `corners` (numpy) | 0.288 s |
| `jacobian` (numpy) | 0.038 s |
| `overlap_triplets` (numba, 11 threads) | 0.154 s |
| `build_operator` (scipy COO to CSR) | 0.454 s |
| `Coadd.add`, 1 / 4 / 8 / 16 planes | 0.21 / 0.57 / 0.85 / 2.93 s |

Peak resident memory for one exposure with 3 variance planes, in a fresh
process: drizzle 1.25 GiB (f32) and 1.66 GiB (f64), spizzle 3.56 GiB.

Findings:

- Goal 3 is not met yet. drizzle already shares the geometry between the
  science and variance planes of one call (~0.05-0.1 s per extra plane),
  so spizzle is ~2× slower for the jwst pattern and faster only than
  separate drizzle passes over the same geometry.
- Only `overlap_triplets` is threaded (~0.70 s on 1 thread, 0.15 s on 11);
  the single-threaded numpy/scipy steps around it take 0.78 s of the
  0.93 s. The COO-to-CSR conversion is not needed: the triplets come out
  in input pixel order, so a CSC array can be built from them without
  sorting.
- `Coadd.add` makes a new array for every product and forms `W∘W`, so
  each plane costs ~0.1 s and the memory peak is ~3× drizzle's.
- gwcs evaluation is not included (synthetic pixel map); measure it with
  the real-data test (§7.4).

### 8.2 After the speed-up, 2026-11-01

Same benchmark, machine and versions as §8.1. Changes:

- `corners` and `jacobian` are numba kernels, parallel over input rows,
  with the same arithmetic as before; the results are bit-identical.
- `build_operator` builds the CSR arrays itself, in parallel: the
  non-zero triplets are counted and bucketed by output row, a chunk of
  triplets per thread, then sorted by column within each row. Both sorts
  keep the triplet order, so the entries of each output pixel stay in
  input pixel order, and the arrays equal scipy's COO-to-CSR result bit
  for bit.
- `Coadd.add` makes one parallel pass over the rows of `W` for the
  science data, weights and context, and one per variance plane, adding
  into its accumulators without temporary arrays or `W∘W`.
- CSR was kept, although CSC needs no sort: the parallel CSR build costs
  0.10 s, and a parallel apply needs CSR rows (gathers); applying a CSC
  array scatters, which does not parallelise without write conflicts.
  Serial CSC apply measured 0.32 s for 4 planes against 0.06 s for the
  parallel CSR one.

| planes | drizzle f32, one call | drizzle f64, one call | drizzle f32, one call per plane | spizzle, `drizzle_operator` + `Coadd.add` | drizzle f32 one call / spizzle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.563 s | 0.600 s | 0.563 s | 0.274 s | 2.06 |
| 4 | 0.722 s | 0.762 s | 2.251 s | 0.318 s | 2.27 |
| 8 | 1.277 s | 0.961 s | 4.502 s | 0.379 s | 3.37 |
| 16 | 2.423 s | 2.295 s | 9.005 s | 0.519 s | 4.67 |

| step | time | baseline |
| --- | ---: | ---: |
| `drizzle_operator`, numba threads 1 / 4 / 8 / 11 | 1.09 / 0.36 / 0.27 / 0.26 s | 1.48 / 0.99 / 0.96 / 0.93 s |
| `corners` | 0.010 s | 0.288 s |
| `jacobian` | 0.003 s | 0.038 s |
| `overlap_triplets` | 0.143 s | 0.154 s |
| `build_operator` | 0.095 s | 0.454 s |
| `Coadd.add`, 1 / 4 / 8 / 16 planes | 0.018 / 0.063 / 0.123 / 0.263 s | 0.21 / 0.57 / 0.85 / 2.93 s |

Peak resident memory with 3 variance planes: spizzle 2.96 GiB (baseline
3.56), drizzle 1.25 GiB (f32) and 1.66 GiB (f64).

Findings:

- Goal 3 is met on this machine: 2.3× faster than drizzle for the
  science data and 3 variance planes in one call, 4.7× for 16 planes. A
  further pass over the same geometry costs ~0.02 s against 0.56 s for a
  drizzle call.
- On one thread spizzle is still slower than drizzle (1.09 s for the
  operator alone against 0.56 s for a whole drizzle call); the speed-up
  comes from the threads. Scaling flattens past 4 threads: the M3 Pro
  has 5 performance cores. A server node would show the scaling better.
- `overlap_triplets` is now 55% of the operator time, and nearly all of
  it is drizzle's `boxer`/`sgarea` (count 0.008 s, fill 0.09-0.11 s,
  compaction 0.027 s). Counting the non-zero overlaps exactly would add a
  second `boxer` pass to save the compaction. A faster clipping
  algorithm would give up bit-for-bit agreement with drizzle.
- Memory is ~2.4× drizzle's. During `build_operator` the triplets (24 B
  per entry), the row buckets (24 B per entry) and the CSR arrays (16 B
  per entry plus 8 B per output pixel) are alive together.

## 9. Open questions / risks

- Memory for `W` on large mosaics: must stream per exposure; consider
  float32 *values* in CSR (geometry precision is not the bottleneck) while
  keeping accumulators float64.
- The CSR `indptr` has one int64 per output pixel, so on a large mosaic
  each exposure's `W` costs 8 bytes per output pixel even where the
  exposure does not land. Restricting `W` to the exposure's bounding box
  on the output grid would avoid that.
- `numba` `parallel=True` with dynamic per-row buffers: may need a two-pass
  (count, then fill) scheme for predictable allocation.
- Rounding helper `nintd` and bounding-box clipping must match the C exactly
  or edge pixels will differ.
- Whether upstream wants (B.2); open an issue early to gauge interest.
- gwcs evaluation cost for corner grids (4× the pixel-centre evaluations) —
  reuse `pixmap` interpolation initially, profile later.

## 10. Suggested first milestone (≈1–2 days)

1. Package skeleton with `corners`, `overlap_triplets` (numba port of
   `boxer`/`sgarea`), `build_operator`, `apply`.
2. Tests 7.1–7.3 passing.
3. Benchmark table for §8 on synthetic data.
4. Decide A vs B and whether to open the (B.1) PR.
