# M4 results — MOTION

`BLOCK_DELTA` can say *that* a region changed. It cannot say a region **moved**, so a rigid
object sliding three pixels right costs the same as that region being repainted at random.
Motion compensation is the fix. **Every configuration verified lossless.**

## Results

Defaults: block size 16, search range 7.

| clip | M3 | M4 | gain | vs raw | modes chosen |
| --- | ---: | ---: | ---: | ---: | --- |
| screen | 2.28 KB | 2.27 KB | 1.00x | 5948x | FULL 1, PIXEL 38, BLOCK 19, MOTION 2 |
| cut | 347.01 KB | 347.01 KB | 1.00x | 38.90x | FULL 3, PIXEL 57 |
| slide | 1.56 MB | **373.54 KB** | **4.27x** | 36.14x | FULL 1, MOTION 59 |
| noise | 7.44 MB | 7.40 MB | 1.01x | 1.78x | FULL 1, MOTION 59 |
| mixed | 400.90 KB | **288.47 KB** | **1.39x** | 46.80x | FULL 1, PIXEL 17, BLOCK 9, MOTION 33 |

Exact bytes: screen 2,324 · cut 355,334 · slide 382,503 · noise 7,755,148 · mixed 295,392.

`slide` is the headline and it is exactly the clip this mode exists for: a textured box
translating 3 px/frame, picked up by `MOTION` on all 59 predicted frames. It is also now
**within 0.4% of the reference implementation running every mode** (380,955 bytes), which is
a useful signal that the search and the residual handling are right.

`mixed` gains 1.39x with `MOTION` winning 33 of 59 frames — its top-right quadrant is a
sliding block, and that is the quadrant `MOTION` was always going to claim.

## Observation 1: one block size cannot serve two modes

M3 predicted this and it is now measured. With `MOTION` available:

| clip | block 8 | block 16 | block 32 | best |
| --- | ---: | ---: | ---: | ---: |
| screen | 2.52 KB | 2.27 KB | **2.09 KB** | 32 |
| cut | 347.01 KB | 347.01 KB | 347.01 KB | — |
| slide | **309.23 KB** | 373.54 KB | 715.17 KB | **8** |
| noise | 7.59 MB | 7.40 MB | **7.33 MB** | 32 |
| mixed | **268.08 KB** | 288.47 KB | 357.78 KB | **8** |

**`slide` swings 2.3x across the range, and the two modes want opposite ends of it.**
`BLOCK_DELTA` wanted 32 at M3, because the unchanged pixels a large block drags in are delta
zeros that zstd erases almost for free. `MOTION` wants 8, because a large block straddling a
moving object's edge cannot be described by one vector and pays a full residual for the whole
block.

Both preferences are correct, for different reasons, and a single `block_size` has to pick
one. On `slide` the default of 16 is costing 20% against block 8; on `noise` block 8 would
cost 3% against 32. That is the clearest argument yet for variable-size partitions — letting
the encoder subdivide where an edge crosses a block rather than choosing one size for the
whole file. It is on the roadmap beyond M10, and this table is why.

The default stays 16 so M5's numbers stay comparable to these. `--block-size 8` is worth
17% on `slide` and 7% on `mixed` today; `--block-size 32` is worth 8% on `screen`.

## Observation 2: the search range costs time, not bytes

On `slide`, which moves exactly 3 px/frame:

| search range | size | encode time |
| ---: | ---: | ---: |
| 1 | 1.55 MB | 1.1s |
| 3 | 373.57 KB | 3.7s |
| **7** (default) | **373.54 KB** | **15.0s** |
| 15 | 373.53 KB | 64.0s |

Range 1 cannot find a 3-pixel displacement and `MOTION` collapses to roughly `BLOCK_DELTA`.
Range 3 finds it. **Everything past 3 costs time and buys 30 bytes** — range 7 is 4x the
time of range 3 for a 0.01% gain, and range 15 is 17x for nothing.

The cost is quadratic by construction: the search visits `(2r+1)²` displacements, so 1 → 3 →
7 → 15 is 9 → 49 → 225 → 961 candidates. The flat size curve says the content's true motion
is 3 and the rest of the window is wasted work.

The catch is that you cannot know the right range in advance — it is a property of the
footage, not of the codec. A clip panning 20 px/frame would need 20 and would collapse like
range 1 does here. So the default stays 7 as an unprincipled compromise, and the honest fix
is a cheaper search rather than a smaller window: that is M9's coarse-to-fine, which gets a
wide window at a fraction of the cost.

## Observation 3: encoding is now 20–200x slower, and the search is all of it

| clip | M3 | M4 |
| --- | ---: | ---: |
| cut | <0.1s | 1.1s |
| screen | <0.1s | 5.2s |
| mixed | <0.1s | 9.1s |
| slide | <0.1s | 13.9s |
| noise | <0.1s | 19.5s |

Two things cause this together. `MOTION`'s `cost()` is the first mode whose size is **not** a
closed form — it needs the search to know which blocks match exactly and therefore carry no
residual, so it encodes and measures. And the `trial` selection strategy prices every
candidate on every frame. So every frame now pays for a full motion search whether or not
`MOTION` wins — on `cut`, which chooses `MOTION` zero times, the search still runs 59 times.

That is the cost structure M6's learned predictor exists to attack: not the search itself but
*pricing modes that are going to lose*. M9 attacks the search's own cost.

## Observation 4: MOTION is not a free upgrade

Each mode restricted to compete only against `FULL`:

| clip | `MOTION` alone | `BLOCK_DELTA` alone | ratio |
| --- | ---: | ---: | ---: |
| screen | 2,480 | 2,369 | **1.05x worse** |
| cut | 355,391 | 355,391 | 1.00x |
| slide | 382,503 | 1,632,803 | **0.23x** |
| noise | 7,755,148 | 7,805,522 | 0.99x |
| mixed | 322,527 | 410,536 | 0.79x |

On `screen`, `MOTION` is **5% worse** than `BLOCK_DELTA`. Its changes are a blinking cursor
and text appearing — things that *appear*, not things that move — so every record pays 3
extra bytes for a `(dx, dy)` of `(0, 0)` plus a flags byte, and gets nothing for them. `noise`
is a wash for the same reason: grain does not move anywhere.

This is why the mode registry is the right shape. `MOTION` is a 4.3x win on translation and a
5% loss on appearance, and the selector resolves that per frame without anyone having to
decide in advance — visible in `screen`'s mode counts, where `MOTION` wins exactly 2 frames
out of 59 and `BLOCK_DELTA` takes 19.

## Where the remaining gap is

| clip | M4 (here) | reference, all modes | gap | what closes it |
| --- | ---: | ---: | ---: | --- |
| screen | 2,324 | 2,267 | 1.03x | M5 `TILED` |
| cut | 355,334 | 231,630 | 1.53x | M8 intra prediction |
| slide | 382,503 | 380,955 | **1.00x** | — essentially closed |
| noise | 7,755,148 | 7,223,174 | 1.07x | M8 intra prediction |
| mixed | 295,392 | 211,974 | 1.39x | M5 `TILED` |

`slide` is done — within 0.4%, and better than the reference at `--block-size 8`. The two
remaining gaps that are not intra prediction are both `TILED`'s: `mixed` at 1.39x, which is
the clip built specifically to need per-region mode choice, and `screen` at 3%.
