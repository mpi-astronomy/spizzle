# drizzle issues found while writing spizzle

Problems in `spacetelescope/drizzle` noticed while porting the square
kernel, to fix upstream later.  Line numbers are for
`jdavies-st/drizzle` branch `float64-phase2-c-accepts-float64` (f37a056).

## 1. Output edge pixels lose weight when the input extends past the output

**Severity:** wrong results (weight and flux lost at the output edges).
**spizzle:** does not reproduce it; decided 2026-11-01 to visit every
input pixel and clip at the output bounds instead.

`init_image_scanner` (`src/cdrizzlemap.c:1089`) clips the input image
outline to the output frame and maps the result back to the input frame;
`get_scanline_limits` (`:919`) then visits only the input pixels whose
centers lie inside that polygon (`init_edge`, `:766`, shifts the edges
inwards by half a pixel, and the limits are rounded).  An input pixel
whose center is just outside the output frame is skipped, although its
footprint covers part of the output edge pixel.

The loss shows at the top, left and right output edges.  The bottom edge
kept its full weight in the cases tried, because the first row comes from
`ymin = (int)(min_y + 0.5 + 2 * MAX_INV_ERR)` rather than from the edge
walk.

Reproduction (uniform data and weights, input pixels the same size as
output pixels, so every covered output pixel should get weight 1):

```python
import numpy as np
from drizzle import cdrizzle

y, x = np.indices((12, 20), dtype=np.float64)
pixmap = np.dstack([x - 9.7, y])   # input column 9 centered at x = -0.7
out = np.zeros((12, 10)); wht = np.zeros((12, 10)); ctx = np.zeros((12, 10), np.int32)
cdrizzle.tdriz(np.ones((12, 20)), np.ones((12, 20)), pixmap, out, wht, ctx,
               pixfrac=1.0, kernel="square", in_units="cps", expscale=1.0,
               wtscale=1.0, fillstr="INDEF")
print(wht[5, 0])   # 0.7, should be 1.0: input column 9 covers [-1.2, -0.2]
```

With `x - 9.4`, output column 9 gets 0.6 instead of 1.0.  A rotated input
that covers a 21x21 output completely leaves 72 (10 deg), 82 (30 deg) and
80 (45 deg) output pixels short of weight 1, all in the outer two rings,
the worst a corner pixel at 0.004.

This matters when the output frame is smaller than the inputs: fixed
output shapes, tiles and sky cells.  The usual jwst output frame encloses
all inputs, so there the polygon is the whole input image.

## 2. Uninitialized scanner fields read when input and output do not overlap

**Severity:** undefined behavior, no visible effect so far.

When the clipped polygon has no vertices, `init_scanner`
(`src/cdrizzlemap.c:803`) returns before setting `s->min_y`, `s->max_y`
and `s->ymax`, and `init_image_scanner` (`:1160-1161`) then computes
`*ymin` and `*ymax` from them.  The kernel returns early because
`init_image_scanner` returned 1, so nothing is drizzled, but the reads
are of uninitialized memory.

## 3. A failed pixmap inversion leaves a partial scan polygon

**Severity:** possibly wrong pixel selection, rare.

If `map_vertex_to_input` fails for vertex k of the clipped polygon
(`src/cdrizzlemap.c:1145`), the code jumps to `_setup_scanner` with
`inpq` holding the k vertices inverted so far, and scans that partial
polygon instead of the input bounding box.

## 4. `max_doubles` starts from the smallest positive double

**Severity:** harmless (extra work only).

`max_doubles` (`src/cdrizzleutil.h:547`) starts from `MIN_DOUBLE`
(2.2e-308) instead of `-MAX_DOUBLE`, so a quadrilateral entirely at
negative x or y gets `max_ii = 0` or `max_jj = 0` instead of a negative
value.  `boxer` then returns 0 for those pixels, so results do not
change.

## 5. float64 results depend on the compiler contracting multiply-adds

**Severity:** last-bit differences between builds; confirmed on Apple
Silicon with clang.

`setup.py` does not pass `-ffp-contract=off`.  With float32 outputs the
effect of fused multiply-adds disappeared in the rounding to float32
(commit de9b7ea); with float64 output arrays, `interpolate_point`,
`interpolate_four_points`, the Jacobian and `sgarea` differ between
builds in the last bits.

Measured 2026-11-01 (macOS arm64, clang default contraction): the output
weights of a single input pixel differ from spizzle's numpy/numba port by
up to 6e-15 on values of order 1, for most edge pixels and some interior
ones.  Built with `CFLAGS=-ffp-contract=off`, the same drizzle gives
weights identical bit for bit to spizzle's, for single pixels and for a
whole image.  Adding `-ffp-contract=off` (or `/fp:precise` on MSVC) to
`extra_compile_args` would make the float64 results reproducible.

## 6. A valid pixmap region one pixel wide is rejected

**Severity:** minor.

`shrink_image_section` (`src/cdrizzlemap.c:124`) returns failure when
`imin >= imax` or `jmin >= jmax`, so a pixmap whose valid pixels form a
single row or column gives "No or too few valid pixels in the pixel
map", and `tdriz` also requires `xmax > xmin` and `ymax > ymin`.

## 7. Blot's valid region is shifted by half a pixel

**Severity:** minor (affects the outermost pixel of the blotted image).
**spizzle:** decided 2026-11-02 to accept the image's extent,
`[-0.5, nx - 0.5)`, holding the edge values beyond the outermost pixel
centers; `blot` matches drizzle bit for bit everywhere else.

Pixel centers are at integer coordinates, so an image of `nx` columns
covers `[-0.5, nx - 0.5)`. `doblot` (`src/cdrizzleblot.c:674`) accepts
positions in `[0, nx)` (and likewise in y). A position in the first half
of the first pixel gets the fill value, while one up to half a pixel
beyond the last edge gets the value of the last column:
`interpolate_bilinear` (`:295`) interpolates there along y only.

Reproduction:

```python
import numpy as np
from drizzle import cdrizzle

data = np.arange(12.0).reshape(3, 4)   # x edges at -0.5 and 3.5
pixmap = np.array([[[-0.25, 1.0], [3.75, 1.0], [1.0, -0.25], [1.0, 2.75]]])
out = np.zeros((1, 4))
cdrizzle.tblot(data, pixmap, out, interp="linear", fillval=-99.0)
print(out)   # [[-99. 7. -99. 9.]]: inside the image filled, outside it given values
```

The same check applies to every interpolation method, not only to the
bilinear one.

## 8. Lanczos blot weights do not sum to one, and come from a truncated table

**Severity:** wrong results for `lanczos3` and `lanczos5` (up to 1%).

`interpolate_lanczos` (`src/cdrizzleblot.c:528`) sums the pixels times
the kernel weights without dividing by the sum of the weights. Lanczos
weights do not add up to one: in 2D the sum ranges over 0.989-1.000 for
`lanczos3` and 0.998-1.000 for `lanczos5`, depending on the position
between pixel centers. The weights also come from a table with spacing
0.003 (`lut_delta`, `:30`; `create_lanczos_lut`,
`src/cdrizzleutil.c:448`), looked up by truncating `|x - i| / 0.003`
rather than by rounding or interpolating. This moves the sums to
0.991-1.008 (`lanczos3`) and 0.999-1.010 (`lanczos5`). At a pixel
center, the neighbors at integer distances get the weight of 0.999,
1.998, ... instead of zero.

Measured with known answers (positions at least 4 pixels from the
edges):

- a flat image is reproduced only to 0.9% (`lanczos3`) and 1.0%
  (`lanczos5`);
- at pixel centers, the result differs from the pixel's value by up to
  5.6e-3 (`lanczos3`) and 7.3e-3 (`lanczos5`) for unit-variance noise;
- `nearest`, `linear`, `poly3` and `poly5` reproduce polynomials up to
  their degree, and pixel values at the centers, to round-off.

Normalizing by the sum of the weights fixes the flat image. Rounding or
interpolating in the table, or computing the kernel directly, fixes the
values at pixel centers.

## 9. Lanczos blot near the edges uses a cut kernel, not the fill value

**Severity:** wrong results within `order` pixels of the edges.

The comment in `interpolate_lanczos` (`src/cdrizzleblot.c:551`) says
that positions close to the edge return the missing value. The code
instead clips the kernel to the image and sums what is left, without
normalizing. For a flat image of ones, `lanczos3` gives 0.79 at
`x = nx - 0.75` and 0.21 at `x = nx - 0.25`. For a linear image the
error is 0.2-0.4 within 3 pixels of the edges. `poly3` and `poly5` use a
reflection about the edge pixel there, and are exact for linear images.

## 10. `poly3` and `poly5` blot of small images aborts or reads outside the image

**Severity:** crash.

`get_pixel_reflect_x` (`src/cdrizzleblot.c:372`) needs the requested
pixel to be less than one image size beyond an edge. For `poly3` on
images smaller than 3 pixels, and `poly5` on images smaller than 4
pixels, in x or y, it is not, and nothing checks the image size before
interpolating. With assertions compiled in, the assertion aborts the
Python process ("Abort trap: 6", seen for `poly3` on 2x2 and `poly5` on
3x3). Without them, it reads outside the image.

## 11. `tblot` accepts `xmin`, `xmax`, `ymin`, `ymax` but ignores them

**Severity:** minor.

`tblot` (`src/cdrizzleapi.c:990`) validates the four limits and stores
them in the parameters (`:1001`). `doblot` then loops over every output
pixel (`src/cdrizzleblot.c:661`). With `xmin=10, xmax=20, ymin=5,
ymax=15` on a 20x30 output, all 600 pixels are set, 479 of them outside
the limits. The arguments should be removed, or used.
