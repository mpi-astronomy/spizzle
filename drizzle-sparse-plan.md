# Plan: faster, float64, parallel drizzle via an explicit sparse operator

**Status:** first milestone done: steps 1-2 in `fc5db6d`, step 3 (benchmarks)
in §8.1 and §8.2, step 4 (A vs B) decided in §6. Real-data check (§7.4)
done, results in §7.1. Next steps (§11): 1-3 done, 4 (exact aperture
variance) planned for later, 5 (in place of drizzle in the jwst steps) in
progress.
**Context date:** 2026-11-01
**Reference code:** `spacetelescope/drizzle` `main` @ `59ff6a1` (2026-10-28)

## 1. Problem statement

`jwst.resample.ResampleStep` is backed by the `drizzle` package. Its C kernel
(`src/cdrizzlebox.c`) is single-threaded, stores and re-normalizes output
pixels in float32 on every contribution, and recomputes the input→output
pixel-overlap geometry for every data plane pushed through it (science, each
variance plane, every blot/outlier-detection iteration, any noise realizations).

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
2. Float64 accumulation; one normalization at the end.
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
  vectorization (branches on clipping). Everything else is array arithmetic.

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
accumulate across exposures in float64; normalize once at end
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
Casertano correction factor. Done as `Covariance`, see §11.3.

## 6. Repository strategy

Two options; pick after the numba prototype works.

**A. Separate package** (recommended for the prototype)
- `pip install`-able, depends on numpy/scipy/numba/gwcs/astropy only.
- Imports `drizzle` solely in the test suite for equivalence checks.
- Fast to iterate; no CI/build-system friction.

**B. Branch of a `drizzle` fork**
- Natural home for two upstreamable pieces once proven:
  1. float64 accumulation + deferred normalization in `update_data_var`
     (change output array dtype, remove per-hit division, normalize in Python
     after the last `add_image`). Small, self-contained, likely acceptable
     upstream.
  2. A new C entry point `boxer_grid(corners, out_shape) -> triplets`
     exposing the geometry without the scatter, so the sparse path can use the
     reference implementation instead of a port.
- Keep the existing API working; add the new path alongside.

Start with A; carry (B.1) as an independent PR regardless.

**Decision, 2026-11-01:** A. spizzle stays a separate package
(mpi-astronomy/spizzle). B.1 is half done in the jdavies-st/drizzle float64
series (`float64-phase1-compute-double`, `float64-phase2-c-accepts-float64`:
double arithmetic and float64 output arrays); the deferred normalization
(sum, then divide once after the last `add_image`) becomes its next phase.
B.2 is dropped: the numba port of `boxer`/`sgarea` matches drizzle bit for
bit (built with `-ffp-contract=off`) and runs in parallel, so a C entry
point would add nothing. The problems in `drizzle-issues.md` go upstream
as separate PRs.

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

### 7.1 Real data, 2026-11-01

`python benchmarks/compare_jwst.py` on the 8 NIRCam `nrcalong` F277W
exposures of program 6154 observation 1 (`~/data/jwst_6154/reduction_mast`,
MAST reduction with jwst 1.17.1), read with gwcs 1.0.3 alone (no jwst).
All 8 go onto one north-up TAN frame enclosing them, by drizzle
(`tdriz`, float32 and float64 outputs) and by spizzle, from the same
`drizzle.utils.calc_pixmap` pixel maps, the image3 default weights
(`ivm`: 1/VAR_RNOISE, zero where DQ has DO_NOT_USE or NON_SCIENCE) and
the VAR_POISSON, VAR_RNOISE and VAR_FLAT planes. Same machine as §8.

Agreement, relative to `max(|drizzle|, median |drizzle|)` over the pixels
with weight:

| frame | output pixels with weight | sci | wht | variance planes |
| --- | ---: | ---: | ---: | ---: |
| native scale, `pixfrac=1`, `cal` | 13.76 M | 4.6e-13 | 8.2e-16 | 4.4e-13 |
| native scale, `pixfrac=1`, `crf` | 13.76 M | 5.3e-14 | 8.2e-16 | 4.2e-13 |
| 2× oversampled, `pixfrac=0.8`, `cal` | 54.95 M | 1.5e-13 | 8.9e-16 | 1.6e-12 |

These are maxima against drizzle with float64 outputs; ~99.99% of pixels
differ by less than 1e-15. Both packages give weight to the same pixels,
NaN in the same pixels, and the same context image. Against drizzle with
float32 outputs (released drizzle) the differences are 1e-9 to 4.9e-7,
float32 round-off. Flux, Σ(sci·wht) / Σ(data·weight): spizzle and float64
drizzle 1 to 15 digits, float32 drizzle off by 4e-11.

Time per exposure, minimum of 3 passes:

| step | native, 5579×4380 | 2× oversampled, 11152×8755 |
| --- | ---: | ---: |
| pixel map (gwcs, `calc_pixmap`) | 0.73 s | 0.72 s |
| spizzle `drizzle_operator` | 0.19 s | 0.39 s |
| spizzle `Coadd.add`, science + 3 variance planes | 0.06 s | 0.14 s |
| drizzle `tdriz`, science + 3 variance planes (f32 or f64) | 0.42 s | 0.68 s |

Findings:

- spizzle is 1.65× faster than drizzle per exposure at native scale and
  1.3× at 2× oversampling. Counting the pixel map, which both need, an
  exposure takes 0.99 s against 1.15 s (native): gwcs evaluation is
  now the largest cost, as §8 expected.
- The operator and `Coadd.add` work on every row of `W`, i.e. every
  output pixel, though one exposure lands on a fraction of a mosaic (4.2 M
  of 24.4 M pixels at native scale). Limiting `W` to the exposure's
  bounding box on the output grid measured 0.27 + 0.06 s instead of
  0.31 + 0.14 s at 2× (one exposure, repeated), and `indptr` 215 MiB
  instead of 745 MiB. See §9.
- With all three packages' 2× outputs in memory (~11 GB of the 18 GB),
  timings rose up to 3× from memory pressure. The script now times each
  package with only its own outputs in memory.

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
  array scatters, which does not parallelize without write conflicts.
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

With the later change to `Coadd.add` from §7.1 (scaling inside the
kernels instead of copying each plane, skipping rows without entries),
`Coadd.add` takes 0.014 / 0.048 / 0.097 / 0.189 s for 1 / 4 / 8 / 16
planes.

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
  on the output grid would avoid that. Measured on real data in §7.1:
  operator plus `add` 30% faster at 2× oversampling, `indptr` 3.5× smaller.
  Done: `Operator`, see §11.
- `numba` `parallel=True` with dynamic per-row buffers: may need a two-pass
  (count, then fill) scheme for predictable allocation.
- Rounding helper `nintd` and bounding-box clipping must match the C exactly
  or edge pixels will differ.
- ~~Whether upstream wants (B.2)~~: dropped, see §6.
- gwcs evaluation cost for corner grids (4× the pixel-center evaluations) —
  reuse `pixmap` interpolation initially, profile later.

## 10. Suggested first milestone (≈1–2 days)

1. Package skeleton with `corners`, `overlap_triplets` (numba port of
   `boxer`/`sgarea`), `build_operator`, `apply`.
2. Tests 7.1–7.3 passing.
3. Benchmark table for §8 on synthetic data.
4. Decide A vs B and whether to open the (B.1) PR.

## 11. Next steps (after the first milestone)

1. **Bounding-box `W`** (§9, measured in §7.1): build each exposure's
   operator on its bounding box on the output grid instead of the whole
   frame, so that the CSR `indptr`, the row sort and `Coadd.add` scale
   with the exposure, not with the mosaic.

   Done, 2026-11-01. `drizzle_operator` and `build_operator` return an
   `Operator`: `matrix`, the CSR array on the box (one row per box
   pixel), `box`, a pair of slices into the output grid (the smallest box
   that holds every non-zero entry), `out_shape`, and `full()` for the
   CSR array on the whole grid, which shares the indices and values.
   `Coadd.add` takes an `Operator` or a sparse array on the whole grid.
   `overlap_triplets` is unchanged; `build_operator` finds the box while
   it counts the entries per output row. `full()` and the `Coadd` outputs
   are bit-identical to the whole-grid version.

   Measured with old and new in one process, alternating (min per
   exposure):

   | case | `drizzle_operator` | `Coadd.add`, 4 planes |
   | --- | ---: | ---: |
   | real, native mosaic 5579×4380 (8 exposures) | 0.201 → 0.190 s | 0.062 → 0.036 s |
   | real, 2× mosaic 11152×8755, one exposure repeated | 0.311 → 0.267 s | 0.137 → 0.057 s |
   | synthetic §8, box ≈ whole frame | 0.287 → 0.297 s | 0.050 → 0.054 s |

   At native scale spizzle now takes 0.23 s per exposure against
   drizzle's 0.43 s (1.9×). `indptr` of the 2× exposure: 745 → 215 MiB.
   When the box is nearly the whole frame the change costs ~3-7%: the
   column range found while counting. A first version that mapped box
   rows to output pixels with an integer division per row made
   `Coadd.add` 30% slower there; the kernels now loop over the box's rows
   and columns instead.
2. **Faster pixel maps.** Think about whether the gwcs pixel map can be
   made faster: at 0.73-0.81 s per NIRCam exposure it now costs more
   than drizzling (§7.1), for drizzle and spizzle alike. Profile of one
   `drizzle.utils.calc_pixmap` call (`nrcalong`, 2048², native-scale
   TAN output frame, 0.81 s):
   - input `pixel_to_world_values` 0.56 s: the four distortion
     polynomials (`multivariate_horner`) 0.26 s, gwcs spherical/Cartesian
     conversions 0.14 s, bounding-box handling 0.06 s, polynomial domain
     mapping 0.04 s;
   - output `world_to_pixel_values` (astropy.wcs TAN) 0.19 s;
   - all of it single-threaded numpy.

   Ideas to weigh, roughly from least to most invasive:
   - evaluate in row blocks on several threads: numpy releases the GIL
     in its array operations, the results are elementwise and so
     identical, and nothing changes for the caller;
   - compute each pixel map once and reuse it in resample, outlier
     detection and blot, which in jwst each evaluate the WCS again;
   - skip the bounding-box machinery when the whole detector is inside
     it;
   - evaluate on a coarse grid (every 8th-32nd pixel) and interpolate:
     the distortion is smooth, but the pixel map then differs slightly
     from the pipeline's, so the interpolation error needs a bound
     (well below 1e-3 pixel) and a test against the full evaluation;
   - compile the transform (numba) for the model types NIRCam imaging
     uses; fast, but tied to one transform structure.

   Prototype, 2026-11-01 (`nrcalong` 2048², against `calc_pixmap`; output
   frame the pipeline's gwcs `i2d` frame, or the astropy TAN mosaic frame
   of §7.1):

   | method | gwcs `i2d` frame | astropy TAN frame | against `calc_pixmap` |
   | --- | ---: | ---: | --- |
   | `drizzle.utils.calc_pixmap` | 0.84 s | 0.80 s | |
   | row blocks of 64 on 8-11 threads | 0.45 s | 0.28 s | bit-identical |
   | input bounding box switched off | no clear gain | no clear gain | bit-identical |
   | every 16th pixel, bicubic spline | 0.08 s | 0.08 s | max 2.1e-8 px |
   | every 32nd pixel, bicubic spline | 0.08 s | 0.07 s | max 3.4e-7 px |
   | every 16th pixel, bilinear | | | max 8.4e-4 px |
   | every 32nd pixel, bilinear | | | max 3.3e-3 px |

   - Threads help less than the cores suggest: gwcs and astropy.modeling
     run many small numpy calls, and the GIL is held between them.
   - The spline is ~10× faster, and bicubic is ~1e4× more accurate than
     bilinear for the same step; nearly all of its time (0.065 s) is the
     spline fit and evaluation on the full grid, not the WCS.
   - This already exists upstream: `stcal.resample.utils.calc_pixmap` has
     `stepsize` and `order` (1 or 3), which jwst's resample and outlier
     detection pass as `pixmap_stepsize` and `pixmap_order`, defaults 1.0
     (no interpolation) and 1 (bilinear). drizzle's own `calc_pixmap` has
     no interpolation; drizzle PR #249 (open) adds `frame_check`, which
     converts between celestial frames through the APE 14 high-level API.
   - Building pixel-map creation into spizzle, on drizzle's function, is
     fine (user, 2026-11-01).

   Done, 2026-11-01: `spizzle.pixel_map(wcs_from, wcs_to, shape=None,
   stepsize=16, order=3, disable_bbox="to", threads=None)`. Bounding boxes
   and shape as in drizzle's `calc_pixmap`; by default the WCS on a grid
   at most 16 pixels apart, bicubic splines between; the full evaluation,
   in row blocks of 64 on numba's thread count, for `stepsize=1` or when
   the grid has a non-finite value. No runtime dependency on drizzle. On
   the 8 `nrcalong` exposures of §7.1: 0.075 s per exposure against 0.84 s
   for `calc_pixmap`, at most 2.2e-8 pixel apart.

   Then the spline evaluated as sparse matrix products, `By @ (C @ Bx.T)`
   with the B-spline basis matrices By and Bx (`order + 1` non-zeros per
   row), instead of `RectBivariateSpline.__call__`: 0.030 s → 0.004 s per
   coordinate on 2048², within 2e-12 of scipy's values, on one thread so
   independent of BLAS threading (dense BLAS took 0.0056 s). Grouping the
   product so that the result comes out in row order, and skipping the
   NaN fill when every pixel is inside the box, removed another 27 ms of
   copying. `pixel_map` now takes 0.022 s per exposure (best of 5; 38×
   faster than `calc_pixmap`), still at most 2.2e-8 pixel from it: ~10 ms
   splines, ~3 ms WCS on the coarse grid, ~8 ms filling the pixel map. An
   exposure through spizzle takes ~0.28 s (pixel map, operator,
   `Coadd.add`) against ~1.27 s for `calc_pixmap` and drizzle.

   Still open:
   - frame conversion: adopt drizzle PR #249's `frame_check` once it is
     merged; until then both WCS must share the celestial frame, as for
     drizzle's current `calc_pixmap`;
   - corners evaluated from the spline instead of bilinear interpolation
     of the pixel map would be more accurate, but no longer
     bit-identical to drizzle (see §5.1).

3. **Noise covariance** (§5.5). Done, 2026-11-01: `Covariance(out_shape,
   radius=2, fillval=nan)`, a separate accumulator next to `Coadd`.
   `add(op, var, iscale=1.0)` takes the same operator and one variance
   image per exposure (e.g. the sum of its variance planes) and keeps its
   own weight sums. `offsets` lists the `(dy, dx)` kept, half of those
   within `radius` (`(0, 0)`, `(0, 1..r)`, `(1..r, -r..r)`: 13 for r=2);
   `out_cov[k][y, x]` is `Cov(img[y, x], img[y + dy, x + dx])`, the sum
   over exposures of `(W diag(var) W.T)[o, o']` over `wsum[o] wsum[o']`.
   Each entry is the dot product of two CSR rows of `W` weighted by the
   variance, merged on their sorted input indices, in parallel over the
   rows of the box; `build_operator` marks its indices as sorted. Offset
   (0, 0) equals `Coadd.out_var` bit for bit. Pixels farther apart than an
   input pixel's footprint share no input and have no covariance, so
   `radius >= ceil(pixfrac * p * sqrt(2))` (p: input pixel size in output
   pixels) gives all of it: 2 at native scale with pixfrac 1, 3 at 2×
   with pixfrac 0.8.

   Checks (`tests/test_covariance.py`, `benchmarks/check_covariance.py`):
   - against the dense `sum W diag(var) W.T / outer(wsum, wsum)` of two
     rotated, weighted exposures: equal to 1e-12 at every offset;
   - 4000 noise realizations through `Coadd` (60×80 input rotated 33°,
     1.6×, pixfrac 0.8, random weights and variances): at each of the 13
     offsets, (sample − predicted) / statistical error over ~12 000
     pixels has mean within ±0.03 and standard deviation 0.99-1.01;
   - noise ratio R of large areas for 16 exposures at random sub-pixel
     offsets, uniform noise, against Casertano et al. (2000),
     `1 / (1 - r/3)` (r ≤ 1) and `r / (1 - 1/(3r))` (r ≥ 1), r = pixfrac
     / s:

     | pixfrac | s | r | R, `Covariance` | R, Casertano |
     | ---: | ---: | ---: | ---: | ---: |
     | 0.5 | 0.5 | 1.000 | 1.523 | 1.500 |
     | 0.5 | 0.75 | 0.667 | 1.304 | 1.286 |
     | 0.5 | 1.0 | 0.500 | 1.196 | 1.200 |
     | 0.75 | 0.5 | 1.500 | 1.947 | 1.929 |
     | 0.75 | 0.75 | 1.000 | 1.512 | 1.500 |
     | 0.75 | 1.0 | 0.750 | 1.328 | 1.333 |
     | 1.0 | 0.5 | 2.000 | 2.400 | 2.400 |
     | 1.0 | 0.75 | 1.333 | 1.778 | 1.778 |
     | 1.0 | 1.0 | 1.000 | 1.495 | 1.500 |

     Within 1.5%; R is the ratio of noise, not of variance. With a single
     exposure and pixfrac < 1 the drops leave uneven weights and the
     formula does not apply (R 1.55 against 1.29 at pixfrac 0.5, s 0.75).
   - NIRCam (one `nrcalong` exposure, radius 2): `Covariance.add` 0.11 s
     on the native mosaic (5579×4380, 13 planes = 2.4 GiB). On the 2×
     mosaic (11152×8755) 0.64 s, but radius 2 is short of the 3 needed
     there and the 13 planes take 9.5 GiB, so the laptop was short of
     memory.

   Use case and decisions (user, 2026-11-01): exposures of 2048² are
   resampled one at a time onto an output grid 2-10× their area; the goal
   is a better estimate of the mosaic's uncertainty. The 12 correlation
   images stay out of the output mosaic, optionally in a separate file; a
   per-pixel noise factor map could replace the ERR array. (Note:
   spizzle's `out_var` is exact for single pixels;
   `sqrt(var * noise_factor)` is right for sums over areas larger than an
   input pixel's footprint, such as apertures, and overstates single-pixel
   errors by `sqrt(noise_factor)`. The pipeline's VAR planes are not the
   exact pixel variance; see "stcal's variance" below.)

   Then, 2026-11-01:
   - `Operator.reach`, the largest distance in y or x between two output
     pixels that share an input pixel, found while `build_operator`
     counts the entries (chunks start at an input pixel's first triplet).
     `Covariance(radius=None)` takes the reach of the first operator and
     warns when one reaches farther than the radius.
   - Sums in float32 by default (each exposure's contribution computed in
     float64); `dtype=np.float64` keeps them, and the variance equal to
     `Coadd.out_var` bit for bit. float32 is within 1e-6 of float64.
   - Results made one image at a time, in parallel: `covariance(k)`,
     `correlation(k)` (the weights cancel: `cnum[o] / sqrt(cnum0[o]
     cnum0[o'])`), and `noise_factor`, each pixel's covariance with all
     pixels over its variance, `wsum[o] / cnum0[o] * sum(cnum[k, o] /
     wsum[o + d] + cnum[k, o - d] / wsum[o - d])`. `out_cov` still stacks
     all images.
   - `write_correlation(path)`: FITS with CORR, a float32 cube of the
     correlation for offsets 1 to n-1, written one image at a time
     (`StreamingHDU`), and an OFFSETS table. astropy is imported only
     there; it is now a test dependency. astropy's `StreamingHDU` takes
     only the name of a `pathlib.Path`, so the path is passed as a string.

   The 8 `nrcalong` exposures one at a time onto the native mosaic
   (5579×4380, 5.8 exposures' area, pixfrac 1, radius 2 from the reach):
   read 0.62 s, `pixel_map` 0.21 s, `drizzle_operator` 4.1 s, `Coadd.add`
   0.51 s, `Covariance.add` 1.18 s in total; `noise_factor` 0.09 s;
   `write_correlation` 1.2 s for 1.09 GiB. Sums: `Coadd` 1.0 GiB,
   `Covariance` 1.37 GiB; peak resident memory 3.2 GiB. Noise factor over
   the 13.8 M pixels with weight: median 2.33, 5% 1.65, 95% 3.38, against
   Casertano's 2.25 everywhere for pixfrac/s = 1.

   Memory for the use case (per output pixel: `Coadd` 44 B, `Covariance`
   (2r(r+1)+1) × 4 + 8 B): native scale, r=2, 2-10× an exposure's area:
   0.5-2.3 GiB for the covariance. 2× oversampling, r=3, 4× the pixels:
   3.4-17 GiB.

   Products at the ResampleStep level (user, 2026-11-01): the
   pixel-to-pixel variance planes as now, plus the noise factor as a
   separate plane, so that photutils can use them together for aperture
   photometry; the correlation images optionally in a separate file.

   How good is the noise factor for apertures? photutils sums `error**2`
   over the aperture, so `error = sqrt(var * noise_factor)` also counts
   each pixel's covariance with its neighbors just outside the aperture.
   Against the exact variance of the aperture sum, `sum(m var) + 2
   sum_k sum_o m(o) m(o + d_k) cov_k(o)` for a mask m of pixel centers
   inside the circle, on the native NIRCam mosaic above (1500 random
   apertures per radius, in variance):

   | aperture radius | var × noise factor / exact: median (5-95%) | var / exact: median | stcal VAR / exact: median (5-95%) |
   | ---: | ---: | ---: | ---: |
   | 1.5 px | 1.27 (1.23-1.35) | 0.56 | 1.23 (1.09-1.29) |
   | 2 px | 1.28 (1.23-1.39) | 0.57 | 1.24 (1.09-1.31) |
   | 3 px | 1.17 (1.10-1.28) | 0.52 | 1.13 (0.98-1.19) |
   | 5 px | 1.09 (1.04-1.17) | 0.48 | 1.05 (0.93-1.11) |
   | 10 px | 1.04 (1.01-1.10) | 0.46 | 0.99 (0.89-1.04) |
   | 20 px | 1.02 (1.01-1.05) | 0.45 | 0.95 (0.88-0.99) |

   The exact pixel variance alone underestimates aperture errors by
   25-33%; with the noise factor they are 13% too large at 1.5-2 px,
   4.5% at 5 px, 1% at 20 px. The excess is the covariance across the
   aperture edge, which depends on the aperture, so no per-pixel plane
   removes it; the exact value needs the correlation images.

   stcal's variance (measured 2026-11-02, same mosaic): `stcal.resample`
   does not propagate the variance through the drizzle weights. For each
   exposure and variance plane it drizzles `sqrt(var)` as an image (a
   weighted mean of the errors) and squares the result. It then combines
   the exposures with weights of one over the resampled read-noise
   variance (or the exposure time), not the drizzle weights. For imaging
   it also divides by the pixel scale ratio squared. Reproduced with
   drizzle on the 8 exposures, stcal's total VAR is 2.23× the exact pixel
   variance (5-95%: 1.55-3.22), and 0.97× the exact variance times the
   noise factor (5-95%: 0.84-1.04). So at native scale with pixfrac 1, the
   pipeline's VAR planes already approximate `var * noise_factor`. As the
   last column shows, for apertures they are close to it, with a wider
   spread, and 5% low at 20 px. Not yet measured: other pixfrac values
   and scales, where the pixel scale ratio factor is all that stcal
   applies.

   The exact correction also varies across the mosaic (distortion, the
   local dither overlap, weights), so one factor per aperture size does
   not fit the whole field. Exact aperture variance over `sum(m var)`, on
   the same apertures:

   | aperture radius | median | 5-95% | min-max | error from the median factor, 5-95% |
   | ---: | ---: | ---: | ---: | ---: |
   | 1.5 px | 1.78 | 1.66-1.89 | 1.43-2.22 | -2.7% to +3.7% |
   | 2 px | 1.77 | 1.64-1.86 | 1.12-2.51 | -2.4% to +4.1% |
   | 3 px | 1.94 | 1.79-2.08 | 1.45-2.53 | -3.5% to +3.9% |
   | 5 px | 2.07 | 1.92-2.21 | 1.36-2.54 | -3.1% to +3.9% |
   | 10 px | 2.18 | 2.05-2.27 | 1.38-2.58 | -2.2% to +3.1% |
   | 20 px | 2.22 | 2.09-2.30 | 1.37-2.50 | -1.7% to +3.2% |

   Open:
   - exact aperture variance at each aperture's position: step 4;
   - 2× oversampled mosaics at the large end: tile the output out of core
     (each exposure's operator to disk, a tile's rows plus a margin of r);
   - covariance for a region only, and the exact variance of apertures or
     other weighted sums through `W.T` in a second pass, if needed.

4. **Exact aperture variance** (to do later; agreed 2026-11-01). Step 3
   shows two things. The noise factor overstates aperture errors by 1-13%,
   depending on the aperture size. The exact correction also varies
   across the field: one factor per aperture size is off by -3.5% to
   +4.1% (5-95%), and by -16% to +27% at the extremes. So each aperture's
   variance has to be computed where the aperture is, from the local
   covariance.

   Proposed function, with no photutils dependency:
   `aperture_variance(variance, correlation, mask, position)`.
   - `variance`: the mosaic's total variance, i.e. the sum of its
     variance planes. This is the variance that `Covariance.add` was
     given, summed over the exposures.
   - `correlation`: the file from `write_correlation`, read through
     astropy's memory mapping, or a `Covariance`.
   - `mask`: the aperture's weights on a cutout, such as photutils'
     `ApertureMask.data` (center, exact or subpixel mode).
   - `position`: `(y0, x0)`, the cutout's first pixel in the mosaic
     (photutils' `mask.bbox.iymin`, `mask.bbox.ixmin`).
   - It returns the variance of `sum(mask * img[cutout])`:
     `sum(m**2 var) + 2 sum_k sum_o m(o) m(o + d_k) c_k(o) s(o) s(o + d_k)`,
     with `s = sqrt(var)` and `c_k` the correlation at offset k.

   Notes for the implementation:
   - Both pixels of each pair lie in the cutout, and `c_k` is stored at
     the first pixel of the pair. So only the cutout of the variance and
     of each correlation plane is read, with no margin.
   - Many apertures: a variant that takes lists of masks and positions
     (photutils' `to_mask()` gives a list). It sorts the apertures by row,
     so that it reads a memory-mapped file in a single pass.
   - Edges and gaps: cutouts partly off the grid are clipped to it. Pixels
     without weight are NaN in the variance and the correlation, and the
     result is NaN when the mask touches one, as in photutils without a
     `mask`. The caller decides what to exclude.
   - Consistency: the variance and the correlation file must come from
     the same exposures and weights. To decide: should `write_correlation`
     also write the total variance, so that the file is self-contained
     and a mismatch cannot happen?
   - Tests: against the dense `m.T C m` of a small rotated case, with
     - a binary mask,
     - a fractional mask,
     - a cutout partly off the grid,
     - the file and a `Covariance` as input, which give the same result.
   - Checks: a Monte Carlo of aperture sums through `Coadd`, added to
     `benchmarks/check_covariance.py`. The tables in step 3 came from a
     scratch script (1500 random apertures per radius on the native
     NIRCam mosaic, all in memory). That measurement also moves into
     `check_covariance.py`, so that it can be rerun.
   - Later: photutils' `aperture_photometry` takes only an `error` array,
     which cannot carry covariance. Whether photutils should accept the
     correlation images is a question for its maintainers, once the
     ResampleStep products are settled.

5. **spizzle in place of drizzle in `ResampleStep` and
   `OutlierDetectionStep`** (agreed 2026-11-02). The caller will live in
   snowblind: a subclass of jwst's `ResampleImage`, the class both steps
   build, plus a replacement for stcal's `gwcs_blot`.

   Where drizzle is called (stcal upstream main 2026-10-07, jwst main
   2026-10-09):
   - `stcal.resample.Resample`, the base of `ResampleImage`: a `Drizzle`
     for the science image (a second one for `compute_err="driz_err"`),
     and for every exposure a new full-frame `Drizzle` for each of its
     three variance planes. Pixel maps come from stcal's `calc_pixmap`.
   - Outlier detection: `stcal.outlier_detection.utils.gwcs_blot` calls
     drizzle's `blot_image` with `interp="linear"`. jwst calls it twice
     per exposure, for the median and for its error, and each call
     computes the pixel map again. That pixel map is the one used to
     resample the exposure onto the median grid.
   - Spectral resampling uses the same stcal class with a 3D WCS; it stays
     on drizzle.

   Decisions (user, 2026-11-02):
   - Weights: inverse variance of the background, zero where DQ has
     DO_NOT_USE, as stcal does. This gives the science mosaic itself the
     best noise; spizzle takes the weights it is given.
   - stcal's VAR and ERR in the mosaic are a rough attempt at the noise
     of sums over areas. They are not reproduced by default. The mosaic
     gets the exact pixel variances (`Coadd.out_var`), the noise factor
     as a separate plane, and ERR = `sqrt(sum of VAR * noise factor)`.
   - A mode that computes stcal's VAR from the operator is useful as a
     sanity check against the stock pipeline, but is not the default.
   - No DQ propagation.
   - No "INDEF" anywhere: fill values are numbers (NaN by default).

   Work in spizzle, in order:
   1. `blot`: bilinear, ported from drizzle's `interpolate_bilinear` and
      `doblot` so that it matches drizzle bit for bit; then a threaded,
      faster version.
   2. Restrict an exposure to a range of input pixels (stcal's
      `xmin`/`xmax`/`ymin`/`ymax` from the bounding box), with an
      interface of its own rather than drizzle's.
   3. stcal's variance from the operator, off by default.
   4. The noise factor in two passes: `Coadd` first, for the weight
      sums, then the covariance of each exposure added straight into
      one noise-factor sum.
   5. Packaging, so that snowblind can depend on spizzle.

   Item 1 done, 2026-11-02: `spizzle.blot(data, pixmap, fillval=nan,
   iscale=1.0, out=None)` in `spizzle/interpolation.py`, bit for bit
   equal to drizzle's `tblot` with `interp="linear"`. This holds for
   float32 and float64 data and output, the last row, column and pixel,
   and positions that are NaN or outside the image. Released drizzle
   2.2.0 interpolates in single precision and differs by up to 1e-5. Each
   pixel is independent, so the rows run in parallel:

   | blot of 2048² from a 5579×4380 float32 median | time |
   | --- | ---: |
   | drizzle `tblot` (C, one thread) | 29 ms |
   | spizzle, one thread | 23 ms |
   | spizzle, 11 threads | 3.7 ms |

   On a real `nrcalong` exposure, as jwst blots in outlier detection
   (median and error, each a `gwcs_blot` call with its own `calc_pixmap`
   of every pixel): 1.61 s. With one `pixel_map` for both and two `blot`
   calls, 0.26 s with `stepsize=1`, bit for bit equal to stcal's, and
   0.029 s with the default splines (at most 2.4e-7 different, about one
   float32 rounding step). The pixel map is nearly all of the cost, so a
   faster interpolation would gain little more.

   Then, 2026-11-02 (user): `blot` accepts the image's extent, `[-0.5,
   n - 0.5)`, instead of drizzle's `[0, n)`, which is half a pixel off
   (`drizzle-issues.md`, issue 7). Beyond the outermost pixel centers it
   holds the edge values, as drizzle already does in its last row and
   column. It still matches drizzle bit for bit wherever both accept the
   position. On the NIRCam exposure above, no blotted pixel changes.
   Checking the other interpolation methods of `tblot` with known answers
   found more problems, logged as issues 8-11: Lanczos weights that do
   not sum to one and come from a truncated table (errors up to 1%),
   Lanczos near the edges, a crash for `poly3` and `poly5` on images of
   2-3 pixels, and ignored `xmin`/`xmax`/`ymin`/`ymax`. stcal uses only
   the bilinear method.

   Then in snowblind: the `ResampleImage` subclass and the `gwcs_blot`
   replacement, and a check against the stock steps on the 8 NIRCam
   exposures, in an environment with jwst and stcal.
