# M1 results — the V2 lossless codec

The first milestone that is a codec. Two competing representations (`FULL` and
`PIXEL_DELTA`), a container, an encoder and decoder, a losslessness proof, and a CLI.

**Every configuration below verified lossless** — every decoded frame byte-identical to the
ffmpeg-decoded source frame, with and without the entropy stage.

## Setup

Five synthetic clips, each built to isolate one behaviour: `screen` (static UI, sparse
edits), `cut` (three scenes, two hard cuts), `slide` (textured box translating), `noise`
(moving gradient plus grain), `mixed` (four quadrants, four behaviours).

All are 320×240, 60 frames, so raw RGB24 is **13.18 MB** (13,824,000 bytes) in every row.
They are encoded `libx264rgb -qp 0`, which is lossless *in RGB* — so the source pixel format
is `gbrp`, not `yuv420p`, and nothing here upsamples chroma. That matters later; see
[ROADMAP.md](ROADMAP.md) item 1 beyond M10.

The strategy at M1 is fixed: frame 0 is `FULL`, every later frame is `PIXEL_DELTA`. No
selection — that is M2.

## Results

| clip | payload (pre-entropy) | final | vs raw frames | changed px/frame | coordinate share |
| --- | ---: | ---: | ---: | ---: | ---: |
| screen | 236.09 KB | **4.21 KB** | 3206.68x | 26 | 56.0% |
| cut | 1.25 MB | **973.75 KB** | 13.86x | 2,603 | 57.1% |
| slide | 3.47 MB | **2.76 MB** | 4.78x | 8,240 | 57.1% |
| noise | 30.43 MB | **18.06 MB** | **0.73x** | 76,709 | 57.1% |
| mixed | 1.90 MB | **760.69 KB** | 17.75x | 4,263 | 57.1% |

Exact bytes, for comparison against later milestones:

| clip | payload | stored | total on disk | raw frames |
| --- | ---: | ---: | ---: | ---: |
| screen | 241,752 | 3,983 | 4,311 | 13,824,000 |
| cut | 1,305,836 | 996,792 | 997,120 | 13,824,000 |
| slide | 3,633,756 | 2,893,660 | 2,893,988 | 13,824,000 |
| noise | 31,911,747 | 18,935,516 | 18,935,844 | 13,824,000 |
| mixed | 1,991,388 | 778,615 | 778,943 | 13,824,000 |

Container overhead is 328 bytes everywhere: a 28-byte header plus 5 bytes per frame record.

Reproduce with:

```
uv run vidc encode bench/clips/cut.mp4 out.myv
uv run vidc verify bench/clips/cut.mp4 out.myv
uv run vidc encode bench/clips/cut.mp4 out-raw.myv --no-entropy
```

## Observation 1: on dense content the file is *larger* than no compression at all

`noise` is **0.73x** — 18.06 MB to describe 13.18 MB of raw frames. Pre-entropy it is worse
still: 30.43 MB, **2.3x larger than the pixels it describes.**

The cause is structural, not a bug. `noise` changes 76,709 of 76,800 pixels per frame, so
`PIXEL_DELTA` spends 7 bytes per pixel to say what storing the pixel outright would say in 3.
A representation chosen for sparse change is being applied to a frame where nothing is
sparse.

This is the entire argument for **M2**. The fix is not a better delta format; it is to stop
committing to one representation in advance. `FULL` was already available on every one of
those frames and would have been cheaper on all of them — the encoder simply never asked.

The same effect, smaller, is visible at `cut`'s two scene boundaries: a hard cut changes
essentially every pixel, and the fixed strategy answers it with a per-pixel delta list.

## Observation 2: four of every seven bytes are coordinates

Measured against the `PIXEL_DELTA` payload, coordinates are **57.1%** on four of five clips —
which is exactly 4/7, the theoretical value, confirming the records are packed with no
padding. (`screen` reads 56.0% only because its payload is small enough for the 4-byte
per-frame count header to show up.)

So more than half of every delta payload is `(x, y)` pairs, and a changed pixel costs the
same 7 bytes whether its colour moved by 1 or by 200. On `cut` that is 600 KB of the 1.25 MB
payload spent on *locations*.

This is the argument for **M3**. Changes are not scattered — they cluster, because moving
objects, cursors, text and UI repaints all dirty contiguous regions. Storing one index per
changed *block* instead of one coordinate pair per changed *pixel* is what `BLOCK_DELTA` does.

## Observation 3: the entropy stage cannot save a bad representation

Pre-entropy against final:

| clip | zstd contribution |
| --- | ---: |
| screen | 60.7x |
| cut | 1.31x |
| slide | 1.26x |
| noise | 1.69x |
| mixed | 2.56x |

On `screen`, zstd does nearly all the work — the payload is mostly a near-empty delta list,
and generic compression eats it. On `cut` and `slide` it recovers barely a quarter, because
coordinate-tagged records are close to incompressible: every `(x, y)` in a frame is distinct
by construction, so there is almost nothing for an LZ matcher to repeat.

That is worth stating plainly because it is the reason the size report prints both numbers.
A single final size would have hidden the fact that on `slide` the representation is doing
all the losing and the entropy coder is doing almost nothing.

## Cross-check against the reference implementation

The prior complete implementation at `../myv-codec` was run at the same configuration
(`--selection fixed --modes FULL,PIXEL_DELTA`) as a correctness check on this rebuild.

**The pre-entropy payloads are byte-identical — to within exactly 1 byte on every clip:**

| clip | this rebuild | reference | difference |
| --- | ---: | ---: | ---: |
| screen | 241,752 | 241,753 | 1 |
| cut | 1,305,836 | 1,305,837 | 1 |
| slide | 3,633,756 | 3,633,757 | 1 |
| noise | 31,911,747 | 31,911,748 | 1 |
| mixed | 1,991,388 | 1,991,389 | 1 |

That single byte is the reference's intra-filter id, which its `FULL` payload carries and
this one does not yet. Every `PIXEL_DELTA` payload matches exactly. The representation is
confirmed correct.

The *final* sizes differ, and the difference is entirely one frame:

| clip | this rebuild | reference | gap | my `FULL` stored | reference `FULL` stored |
| --- | ---: | ---: | ---: | ---: | ---: |
| screen | 4,311 | 4,251 | +60 | 216 | 156 |
| cut | 997,120 | 873,385 | **+123,735** | 123,806 | **71** |
| slide | 2,893,988 | 2,894,003 | **−15** | 187,359 | 187,374 |
| noise | 18,935,844 | 18,851,667 | +84,177 | 204,219 | 120,042 |
| mixed | 778,943 | 731,059 | +47,884 | 163,648 | 115,764 |

On `cut` the gap closes to the byte: 123,806 − 71 = 123,735, which is exactly the total
difference. The reference's `FULL` frame is **1,744x smaller than mine** on that clip.

The cause is intra prediction, which this rebuild schedules at **M8**. `cut`'s first frame is
a smooth gradient: raw interleaved RGB of a gradient is nearly incompressible to an LZ coder,
because consecutive bytes keep changing. De-interleave the channels and store each pixel as
its difference from its neighbour and the same frame becomes a run of identical small
numbers, which zstd reduces to 71 bytes.

**This quantifies what M8 is worth, before it is built: 124 KB on a single frame of a
single clip, and 249.8 KB across the five.**

It is worth being precise about how big a lever that is, because it is tempting to overstate.
On *these* clips it is not the biggest: the reference's `MOTION` mode alone takes `slide` from
1.56 MB to 372 KB, a saving of 1,225 KB, five times what intra prediction is worth here. What
the reference measured was that intra prediction won by more than the mode work *on real
1080p footage* — 1.51x — because real frames are dithered and textured and almost never
predict well temporally, so `FULL` is chosen far more often there than on synthetic clips.
Both findings are true of their own content, which is the argument for measuring against real
footage before trusting any ordering.

One row disagrees, and it is instructive: on `slide` this rebuild is **15 bytes smaller**.
The reference's filter chooser picks by an order-0 entropy proxy rather than by trial
compression, and on `slide` it chose a filter that made the payload 12 bytes worse, then paid
1 more byte to record which filter it had used. Intra prediction is not a universal win, and
the chooser is a heuristic — something to keep in mind at M8 rather than discover there.

## What M1 does not do

Named here so the next milestone's numbers have something to improve on, and so nothing
below reads as an oversight:

- **No mode selection.** The strategy is fixed. M2.
- **No spatial grouping.** Every changed pixel pays its own coordinates. M3.
- **No motion compensation.** A translating object is re-described from scratch. M4.
- **No intra prediction.** `FULL` writes raw pixels. M8, and Observation 3 prices it.
- **No keyframes or index.** Seeking means decoding from frame 0. M7.
