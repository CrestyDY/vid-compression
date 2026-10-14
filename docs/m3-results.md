# M3 results — BLOCK_DELTA

Store whole changed *blocks* instead of per-pixel coordinates, attacking the 57.1% of every
`PIXEL_DELTA` payload that M1 measured as `(x, y)` pairs. **Every configuration verified
lossless.**

## Results

| clip | M2 | M3 | gain | vs raw | modes chosen |
| --- | ---: | ---: | ---: | ---: | --- |
| screen | 4.21 KB | **2.28 KB** | **1.85x** | 6060x | FULL 1, PIXEL 38, BLOCK 21 |
| cut | 347.01 KB | 347.01 KB | 1.00x | 38.90x | FULL 3, PIXEL 57 |
| slide | 2.76 MB | **1.56 MB** | **1.77x** | 8.47x | FULL 1, BLOCK 59 |
| noise | 11.70 MB | **7.44 MB** | **1.57x** | 1.77x | FULL 1, BLOCK 59 |
| mixed | 760.69 KB | **400.90 KB** | **1.90x** | 33.67x | FULL 1, PIXEL 17, BLOCK 42 |

Exact bytes: screen 2,331 · cut 355,334 · slide 1,632,803 · noise 7,805,522 · mixed 410,519.

Four of five clips improved — the broadest milestone so far, because coordinate overhead was
a tax on every delta frame rather than a weakness of particular content.

`cut` is unchanged, and the mode counts say why: it picks `PIXEL_DELTA` on all 57 predicted
frames. Its scenes are *static* between cuts, so almost nothing changes and a near-empty
coordinate list is already the cheapest possible answer. `BLOCK_DELTA` has nothing to win
where there is no clustering to exploit.

## Observation 1: the proxy broke, exactly as predicted

M2 recorded a prediction: `trial` and `actual` agreed on all 300 frames then, but
`BLOCK_DELTA` would break it, because it emits a plane of raw delta *values* (low-entropy)
against `PIXEL_DELTA`'s near-incompressible coordinate records. Measured now:

| clip | `actual` | `trial` | disagreements | cost of ranking on the proxy |
| --- | ---: | ---: | ---: | ---: |
| screen | 4,311 | **2,331** | 21/60 | **+84.9%** |
| cut | 355,334 | 355,334 | 0/60 | identical |
| slide | 1,632,803 | 1,632,803 | 0/60 | identical |
| noise | 12,266,566 | **7,805,522** | **59/60** | **+57.2%** (+4.46 MB) |
| mixed | 410,611 | 410,519 | 2/60 | +0.0% |

On `noise` the two strategies disagree on **59 of 60 frames**, and ranking on pre-entropy
size costs 4.46 MB — it would have shipped a file 57% larger while believing it had chosen
optimally. `actual` on `noise` is byte-identical to M2's total, meaning it never picks
`BLOCK_DELTA` at all: pre-entropy, a full plane of delta bytes looks no better than a
coordinate list, so the proxy cannot see the mode that wins by 1.57x after compression.

This is the clearest result in the project so far about *measurement* rather than
compression. **A proxy metric stays valid only while the thing it proxies for is distributed
the way it was when you adopted it.** Two modes agreed for 300 frames; the third broke the
agreement on 59 of 60. Nothing in the code would have complained — the file would just have
been bigger, and the strategy would have looked fine.

Keeping `actual` is what made this measurable instead of a story.

## Observation 2: the default block size is wrong

| clip | 4 | 8 | **16** | 32 | 64 | best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screen | 2.75 KB | 2.52 KB | 2.28 KB | **2.09 KB** | 2.14 KB | 32 |
| cut | 347.01 KB | 347.01 KB | 347.01 KB | 347.01 KB | 347.01 KB | — |
| slide | 1.64 MB | 1.57 MB | 1.56 MB | **1.55 MB** | 1.56 MB | 32 |
| noise | 8.25 MB | 7.62 MB | 7.44 MB | 7.38 MB | **7.35 MB** | 64 |
| mixed | 424.53 KB | 407.38 KB | 400.90 KB | **398.49 KB** | 398.75 KB | 32 |

16 is the inherited default and it is not optimal anywhere: **32 wins on three clips and 64
on `noise`.** The gains are modest (1–9%), and the curve is shallow and flat near the
optimum, which is why 16 is defensible rather than wrong.

The reason larger is better here is that the unchanged pixels a big block drags in are delta
*zeros*, and zstd erases runs of zeros almost for free — so the index saving is real while
the padding cost is mostly illusory. That logic will not survive M4: `MOTION` wants *small*
blocks, because a large block straddling a moving object's edge cannot be described by one
vector and pays a full residual. One `block_size` already serves two modes that want
opposite things, which is the first sign of the variable-size-partition work the reference's
roadmap lists.

The default stays 16 for now, so M4's numbers are comparable to these. `--block-size 32` is
a free 1–9% today for anyone who wants it.

## Observation 3: BLOCK_DELTA beats PIXEL_DELTA almost everywhere, alone

Each mode restricted to compete only against `FULL`, at block size 16:

| clip | `BLOCK_DELTA` alone | `PIXEL_DELTA` alone | ratio |
| --- | ---: | ---: | ---: |
| screen | 2,369 | 4,311 | **0.55x** |
| cut | 355,391 | 355,334 | 1.00x |
| slide | 1,632,803 | 2,893,988 | **0.56x** |
| noise | 7,805,522 | 12,266,566 | **0.64x** |
| mixed | 410,536 | 778,943 | **0.53x** |

`BLOCK_DELTA` is roughly twice as good as `PIXEL_DELTA` on four clips — so one might ask why
keep `PIXEL_DELTA` at all. Two answers. On `cut` it is marginally better and wins all 57
predicted frames. And on `screen` and `mixed` the *mixture* beats either alone: screen ships
2,331 with both available against 2,369 with only blocks, and mixed picks `PIXEL_DELTA` on 17
frames and `BLOCK_DELTA` on 42. The modes are complements, not competitors — which is the
design's whole premise, now visible in the mode counts rather than asserted.

## Where the remaining gap is

| clip | M3 (here) | reference, all modes | gap | what closes it |
| --- | ---: | ---: | ---: | --- |
| screen | 2,331 | 2,267 | 1.03x | M5 `TILED` — nearly closed |
| cut | 355,334 | 231,630 | 1.53x | M8 intra prediction |
| slide | 1,632,803 | 380,955 | **4.29x** | **M4 `MOTION`** |
| noise | 7,805,522 | 7,223,174 | 1.08x | M8 intra prediction |
| mixed | 410,519 | 211,974 | 1.94x | M4 `MOTION` + M5 `TILED` |

`screen` is within 3% of the full reference and `noise` within 8%. `slide` remains the
outlier at 4.29x — a translating textured box, still described from scratch every frame by a
mode that has no concept of movement. That is M4.
