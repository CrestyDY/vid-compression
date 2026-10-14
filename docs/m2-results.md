# M2 results — mode selection

M1 committed every frame to one representation chosen in advance. M2 prices every available
mode per frame and keeps the cheapest. **Every configuration verified lossless.**

## Results

Final on-disk size. Same five clips, 320×240, 60 frames, raw RGB24 = 13.18 MB.

| clip | M1 (`fixed`) | M2 (`trial`) | improvement | vs raw frames | modes chosen |
| --- | ---: | ---: | ---: | ---: | --- |
| screen | 4.21 KB | 4.21 KB | 1.00x | 3206.68x | FULL 1, PIXEL_DELTA 59 |
| cut | 973.75 KB | **347.01 KB** | **2.81x** | 38.90x | FULL 3, PIXEL_DELTA 57 |
| slide | 2.76 MB | 2.76 MB | 1.00x | 4.78x | FULL 1, PIXEL_DELTA 59 |
| noise | 18.06 MB | **11.70 MB** | **1.54x** | **1.13x** | FULL 60 |
| mixed | 760.69 KB | 760.69 KB | 1.00x | 17.75x | FULL 1, PIXEL_DELTA 59 |

Exact bytes: `cut` 997,120 → 355,334. `noise` 18,935,844 → 12,266,566.

```
uv run vidc encode bench/clips/cut.mp4 out.myv                      # trial, the default
uv run vidc encode bench/clips/cut.mp4 m1.myv --selection fixed     # M1's behaviour
uv run vidc encode bench/clips/cut.mp4 f.myv --modes FULL           # one mode alone
```

## Observation 1: the file is no longer larger than no compression at all

`noise` was M1's embarrassment — **0.73x**, an 18.06 MB file describing 13.18 MB of raw
frames. It is now **1.13x**, and the mechanism is worth stating because it required no new
mode at all:

**Selection chose `FULL` on all 60 frames.** `FULL` already existed in M1; the encoder simply
never asked whether it was cheaper. `noise` changes 76,709 of 76,800 pixels per frame, so
`PIXEL_DELTA` was spending 7 bytes per pixel to say what 3 bytes says outright, on every
frame, because the strategy had been fixed in advance.

This is the clearest result in the project so far: **the win came from removing a decision,
not from adding a capability.**

## Observation 2: selection finds the scene cuts without being told about them

`cut` improves 2.81x, and the mode counts say exactly why: `FULL 3, PIXEL_DELTA 57`. The clip
has three scenes, and selection picked `FULL` precisely three times — once per scene start,
where nearly every pixel changes and a per-pixel delta list is the wrong answer.

Nothing in the encoder knows what a scene cut is. There is no detector, no threshold, no
heuristic about frame similarity. Pricing both representations and keeping the smaller one
finds the cuts as a side effect of arithmetic. That is the argument for the whole
competing-representations design, and it is the first milestone where it pays.

## Observation 3: three clips did not move at all, and that is the expected result

`screen`, `slide` and `mixed` are byte-identical to M1. Each is spatially homogeneous —
whatever kind of change it has, it has everywhere — so a single frame-level choice was
already the right choice on every frame, and there was nothing for selection to win.

That is the mode registry working as designed rather than a disappointment: each milestone
should help the clips whose weakness it addresses and leave the others alone. A change that
improved everything equally would more likely be a bug in the measurement.

`slide` in particular needs `MOTION` (M4) and cannot be helped before it: a translating
textured box is described badly by both of the representations that currently exist.

## Observation 4: only one clip benefits from *mixing* modes

There are two different things selection can do, and separating them changes what M2 is
worth. Against the best the file could have done locked to a single mode for its whole
length (frame 0 forced to `FULL`, since it has nothing to predict from):

| clip | best single mode | choosing per frame | gain from mixing |
| --- | ---: | ---: | ---: |
| screen | 3.89 KB | 3.89 KB | 1.00x |
| cut | 973.43 KB | **346.69 KB** | **2.81x** |
| slide | 2.76 MB | 2.76 MB | 1.00x |
| noise | 11.70 MB | 11.70 MB | 1.00x |
| mixed | 760.37 KB | 760.37 KB | 1.00x |

**Only `cut` gains from varying the mode within a file.** On the other four, one mode wins
every single frame, so the per-frame choice lands on the same answer throughout.

That reframes `noise`'s 1.54x: selection did not *mix* anything there, it picked a better
**single** mode than M1's fixed strategy had. Both are wins, but they are different wins, and
only the first is an argument for per-frame selection as such. The second is an argument
against hardcoding a choice — which is the cheaper lesson and the one M1 paid for.

`cut`'s 2.81x is the genuine article: three `FULL` frames interleaved with 57 deltas, a
combination no single mode can express.

*(This table also caught a bug in the size report. The first version compared each mode's
total against the whole file without accounting for frame 0 being unavailable to predicted
modes, which made selection appear to **lose** to a single mode on three clips — impossible,
since the chosen payload is the per-frame minimum. A test now pins the invariant.)*

## Observation 5: `trial` and `actual` agree on every frame — for now

The two strategies rank candidates differently: `actual` compares pre-entropy payload sizes,
`trial` compares sizes after zstd. Measured across all 300 benchmark frames:

| clip | `actual` | `trial` | disagreements |
| --- | ---: | ---: | ---: |
| screen | 4,311 | 4,311 | 0/60 |
| cut | 355,334 | 355,334 | 0/60 |
| slide | 2,893,988 | 2,893,988 | 0/60 |
| noise | 12,266,566 | 12,266,566 | 0/60 |
| mixed | 778,943 | 778,943 | 0/60 |

**Identical, everywhere.** So `trial`'s extra zstd pass per candidate per frame currently
buys nothing measurable, and `actual` is strictly cheaper to compute.

`trial` is still the default, and the reason is worth being precise about rather than
hand-waving. Pre-entropy size is a valid ranking signal only while all candidates have
*similar compressibility*. With two modes that both emit uncompressed-ish bytes, they do.
The failure is easy to construct even with only these two modes — on a 32×24 frame holding
a solid background and a small white square:

```
pre-entropy :  FULL 2304   PIXEL_DELTA 228     -> PIXEL_DELTA looks 10x better
post-entropy:  FULL   36   PIXEL_DELTA 104     -> FULL actually ships 2.9x smaller
```

A nearly-flat frame compresses to almost nothing, while 228 bytes of coordinate records —
every `(x, y)` distinct by construction — barely compress at all. Rank on the proxy and you
ship the 104-byte payload believing it is the 228-byte one.

The benchmark clips simply do not contain that case. **M3 will.** `BLOCK_DELTA` emits a plane
of raw delta *values*, which is low-entropy, against `PIXEL_DELTA`'s near-incompressible
coordinate records — two payloads of identical size can differ 2x after entropy coding. The
disagreement count in the size report is the measurement that will catch it, and keeping
`actual` is what makes the comparison possible instead of asserted.

**The general lesson:** a proxy metric stays valid only while the thing it proxies for is
distributed the way it was when you adopted it. Adding a candidate with a different
statistical shape can silently break it, and nothing in the code will complain.

## Where the remaining gap is

Against the reference implementation running every mode on the same clips:

| clip | M2 (here) | reference, all modes | gap | what closes it |
| --- | ---: | ---: | ---: | --- |
| screen | 4,311 | 2,267 | 1.90x | M3 `BLOCK_DELTA`, M5 `TILED` |
| cut | 355,334 | 231,630 | 1.53x | **M8 intra prediction** (≈123,700 bytes, one frame) |
| slide | 2,893,988 | 380,955 | **7.60x** | **M4 `MOTION`** |
| noise | 12,266,566 | 7,223,174 | 1.70x | **M8 intra prediction** (≈84 KB per FULL frame) |
| mixed | 778,943 | 211,974 | 3.67x | M4 `MOTION` + M5 `TILED` |

Two of the five gaps are now entirely intra prediction, which is consistent with what M1
measured directly: `cut`'s gap of 123,704 bytes is within 31 bytes of the 123,735 the M1
cross-check attributed to the filter on its single `FULL` frame. The clips that chose `FULL`
more often at M2 inherited more of that gap — which is the first sign that M8's value grows
as selection gets better at choosing `FULL`.

`slide` is the outlier and the clearest signpost: 7.6x, untouched by anything so far, waiting
on motion compensation.
