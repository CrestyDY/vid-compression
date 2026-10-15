# Roadmap

This project is a from-scratch rebuild of an experimental lossless video codec. A prior
complete implementation of the same idea exists at `../myv-codec` (V7, ~4,200 lines, 474
tests) and is used here as a **reference for sequence and for measured targets** — not as
code to copy. This rebuild makes different architecture choices on purpose:

| | reference (`myv-codec`) | this rebuild |
| --- | --- | --- |
| package root | `myv/` | `src/` |
| data objects | frozen dataclasses, beside their logic | **pydantic** `BaseModel(frozen=True)`, collected in `src/models/` |
| serialisation | `Header.pack()` methods on the model | packing functions in `src/format.py`, models stay pure |

## The guiding idea

**No single representation of change is always optimal.** Sometimes the cheapest way to
describe frame N is a handful of pixel deltas; sometimes it's "this block moved 20 pixels
right"; sometimes it's storing the whole frame again. A codec is best understood as a
collection of competing representations, with an encoder that prices them and keeps the
smallest.

Reconstruction is **lossless** at every milestone. That is the invariant: every decoded
frame must be byte-identical to the ffmpeg-decoded source frame. A milestone that improves
size but breaks this has failed.

## Milestones

Each milestone answers a weakness the previous milestone's numbers exposed. That ordering is
the point — the measurements are what motivate the next piece of work.

| # | milestone | delivers | reference target |
| --- | --- | --- | --- |
| ~~**M0**~~ | ~~Foundations~~ | deps, test harness, `probe`, frame streaming over ffmpeg pipes | **done** — video I/O layer, round-trip exact |
| ~~**M1**~~ | ~~V2 codec~~ | `FULL` + `PIXEL_DELTA`, container, encoder/decoder, `verify`, CLI | **done** — [results](m1-results.md) |
| ~~**M2**~~ | ~~V3 selection~~ | price every mode per frame, keep the smallest | **done — 2.81x on `cut`**, and `noise` no longer exceeds raw ([results](m2-results.md)) |
| **M3** | V4 `BLOCK_DELTA` | whole changed blocks, no per-pixel coordinates | up to 2.4x (`slide` 2.76 MB → 1.56 MB) |
| ~~**M4**~~ | ~~V5 `MOTION`~~ | block matching + residuals | **done — 4.27x on `slide`**, within 0.4% of the full reference ([results](m4-results.md)) |
| ~~**M5**~~ | ~~V6 `TILED`~~ | an N×N grid, each tile choosing its own mode | **done — 1.16x on `mixed`**, `screen` ahead of the reference ([results](m5-results.md)) |
| **M6** | V7 predictor | learned mode prediction instead of trial encoding | 1.4–12x faster encode, some size cost |
| **M7** | Keyframes | forced `FULL` every N frames + a seek index | seeking bounded by interval, not position |
| **M8** | Intra prediction | de-interleave channels, store spatial differences | **1.51x on real footage** |
| **M9** | Coarse-to-fine search | search decimated, refine, propagate vectors | 2–3x faster for ≤0.5% larger |
| **M10** | MV seeding + player | seed search from source H.264 vectors; direct playback | 5.25x faster search; negligible size |

Benchmark figures are 320×240, 60 frames, raw RGB24 = 13.18 MB, measured on five synthetic
clips (`screen`, `cut`, `slide`, `noise`, `mixed`) that each isolate one behaviour.

### A note on M8's position

The reference added intra prediction **last**, and its own notes call it "the biggest thing
missing" — worth 1.51x on real footage, more than M3, M4 and M5 combined, because it is not
a mode at all and improves every pixel-carrying payload at once.

It is kept at M8 here deliberately. Moving it earlier would produce a smaller file sooner
but would obscure *why* each mode exists: M3 is only obviously necessary once M2's numbers
show what per-pixel coordinates cost. The sequence is the lesson.

## Beyond M10

The reference's forward roadmap, once this rebuild catches up. Recorded here so the
direction is known, with one correction already made to it:

1. **Source-native pixel format.** Real footage is `yuv420p`; decoding to RGB24 *upsamples*
   chroma 4x, so the codec spends bytes losslessly preserving colour detail the file never
   contained. Half the data through every stage.

   The reference frames this as "switch to a YUV420 pipeline". That is wrong as stated:
   measured here, the synthetic benchmark clips are `gbrp` (true lossless RGB, from
   `libx264rgb -qp 0`), and forcing one through yuv420p corrupts **94.2% of bytes, max error
   250**. A global switch would break losslessness on all five clips. The fix is to make the
   internal format *follow the source* — `gbrp` → RGB24, `yuv420p` → planar YUV420, never
   converting — which keeps every clip lossless and makes the claim stronger: byte-identical
   to the source's own samples, in the source's own colour space.

2. **Variable-size motion partitions.** A block straddling a moving object's edge cannot be
   described by one vector, so it pays a full residual. Three independent findings point
   here: no single block size serves both block-modes and motion, no single grid serves all
   content, and residuals are 97–99% of the `MOTION` payload on every clip.
3. **A real entropy coder.** Once payloads are intra-predicted their symbol distributions are
   strongly non-uniform — what a context-modelled range coder exploits and zstd cannot.
4. **Sub-pixel motion compensation.** Worth measuring only *after* item 1: at 2.3% exact
   block matches on real footage, the residual may be dither-limited rather than
   prediction-limited.
5. **Cost prediction instead of mode prediction.** Pricing `MOTION` and `TILED` is 93% of
   what selecting a frame costs, and prediction only pays to the extent the losers were
   expensive.

The goal was never to beat H.264. It's to find out, from the inside, why H.264 is shaped the
way it is.
