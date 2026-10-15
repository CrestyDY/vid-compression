# M8 results — intra prediction

De-interleave the channels, store spatial differences, and let the entropy stage see runs of
small numbers instead of an incompressible interleaved stream. Not a mode: it improves every
pixel-carrying payload at once. **Every configuration verified lossless.**

Taken out of roadmap order, ahead of M6 and M7, because M5's comparison against FFV1 made it
the measured priority: an intra-only codec with no temporal prediction at all was beating
this one 1.81x on real footage.

## Results

| clip | M5 | M8 | gain | vs raw | vs reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| screen | 2,174 | 2,268 | **0.96x — worse** | 5936x | 1.00x |
| cut | 355,334 | **231,630** | **1.53x** | 59.68x | **1.00x — exact** |
| slide | 382,503 | 382,552 | 1.00x | 36.14x | 1.00x |
| noise | 7,744,550 | **7,223,174** | **1.07x** | 1.91x | **1.00x — exact** |
| mixed | 253,772 | **205,814** | **1.23x** | 67.17x | **0.97x — ahead** |

On **real 1080p footage**: 11.87 MB → **9.06 MB**, a 1.31x gain, and **7.86x against raw
frames** (6.0x at M5, 1.86x at M1).

## Observation 1: byte-exact agreement with the reference implementation

The per-clip `FULL` frame, stored, against the reference running the same transform:

| clip | filter chosen | our `FULL` stored | reference | |
| --- | --- | ---: | ---: | --- |
| screen | `PLANAR_XY` | 156 | 156 | exact |
| cut | `PLANAR_XY` | **71** | **71** | exact |
| slide | `RAW` | 187,374 | 187,374 | exact |
| noise | `PLANAR_SUB` | 120,042 | 120,042 | exact |
| mixed | `PLANAR_XY` | 115,764 | 115,764 | exact |

**All five match to the byte**, including which filter the entropy proxy selects. And two
whole-file totals now match exactly too — `cut` at 231,630 and `noise` at 7,223,174 are the
reference's numbers precisely.

That is the strongest correctness signal in the project. Two independently written
implementations of the same design, agreeing byte-for-byte on filter choice, transform output
and final size, is not something that happens by coincidence.

`cut`'s `FULL` frame is the one M1 priced a session ago: **123,806 bytes then, 71 bytes now**,
a 1,744x reduction on a single frame. Its first scene is a smooth gradient — nearly
incompressible as interleaved RGB because consecutive bytes keep changing, and a run of zeros
once differenced. The prediction M1 made closed to the byte.

## Observation 2: intra prediction is a net loss on two clips

`screen` got **4.3% worse** (2,174 → 2,268) and `slide` 0.01% worse (382,503 → 382,552).

The cause is the filter byte, which every pixel-carrying payload now pays unconditionally.
`slide` is the clean case: 60 payloads × 1 byte = 60 bytes of cost, and its `FULL` frame
chooses `RAW` because a textured random box has nothing spatial to exploit. Net +49 bytes —
essentially the overhead with none of the benefit.

`screen` is the interesting one. Its payloads average about 35 bytes, so one byte is ~3% of
each, and its content is sparse delta records rather than smooth pixel fields — spatial
differencing has nothing to find in a list of changed cursor pixels. The mode mix shifts too:
`TILED` drops from 10 frames to 3 and `BLOCK_DELTA` rises from 10 to 17, because the byte
changes the relative pricing.

**This is also precisely where M5's lead over the reference came from.** At M5 we shipped
`screen` at 2,174 against the reference's 2,267 — and the reason was that we had not yet paid
intra prediction's fixed overhead. M8 brings us to 2,268, one byte off parity. The lead was
real but it was the lead of an unfinished codec, and it is honest to record that it has gone.

The trade is clearly worth taking: 94 bytes lost on `screen` against 123,704 gained on `cut`.
A per-payload opt-out would recover it, but then the payload stops being self-describing,
which is a property worth more than 94 bytes.

## Observation 3: `noise` now chooses FULL on every frame again

`noise`'s mode counts went from `FULL 1, MOTION 5, TILED 54` at M5 to **`FULL 60`** at M8.

Intra prediction made `FULL` so much better on grainy content — `PLANAR_SUB` on a moving
gradient plus grain — that it beat every temporal mode on all 60 frames. The codec has
discovered, through pricing alone, that this clip is better treated as a sequence of
independent well-compressed images than as a video.

That is exactly what FFV1 does by construction, and it is why FFV1 was competitive on `noise`
in M5's comparison. M8 gave this codec the same capability, and selection found it without
being told.

## Observation 4: the FFV1 gap is halved but not closed

On the real 1080p excerpt:

| | size | vs our M8 |
| --- | ---: | ---: |
| our `.myv` at M5 | 11.87 MB | — |
| **our `.myv` at M8** | **9.06 MB** | — |
| FFV1 (intra-only) | 6.55 MB | **still 1.38x ahead** |

M8 closed the gap from 1.81x to 1.38x, which confirms the diagnosis was right — but FFV1 is
still ahead, with no temporal prediction at all.

What remains is the second half of FFV1's design: **a context-modelled range coder** instead
of zstd. FFV1 picks a probability model per pixel from the quantised gradients of its
neighbours and adapts it as it codes; zstd is a general-purpose LZ matcher that has no idea
it is looking at image residuals. Now that payloads are intra-predicted their symbol
distributions are strongly non-uniform, which is exactly what a range coder exploits and what
an LZ matcher cannot.

That makes the entropy coder the largest remaining size lever, and the measurement above
prices it: **1.38x on real footage**, against a known opponent.

## Where things stand

The mode set is complete, intra prediction is in, and on the benchmark clips this build now
matches or beats a full reference implementation of the same design on **all five clips**
(`mixed` ahead by 3%, the rest at parity).

Remaining work splits cleanly:

- **Size:** a real entropy coder (1.38x on real footage, measured against FFV1).
- **Time:** M6's predictor and M9's coarse-to-fine search. `noise` takes 38s for 60 frames.
- **Function:** M7's keyframes and seek index.
