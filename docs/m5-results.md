# M5 results — TILED

Every mode before this one commits the *whole frame* to a single representation. A frame can
easily have a static top half and a moving bottom half, and the frame-level selector has to
pick one answer for both. `TILED` splits the frame into a grid and lets each tile choose
independently. **Every configuration verified lossless.**

## Results

Defaults: block size 16, search range 7, grid 4.

| clip | M4 | M5 | gain | vs raw | modes chosen |
| --- | ---: | ---: | ---: | ---: | --- |
| screen | 2.27 KB | **2.12 KB** | **1.07x** | 6359x | FULL 1, PIXEL 38, BLOCK 10, MOTION 1, TILED 10 |
| cut | 347.01 KB | 347.01 KB | 1.00x | 38.90x | FULL 3, PIXEL 57 |
| slide | 373.54 KB | 373.54 KB | 1.00x | 36.14x | FULL 1, MOTION 59 |
| noise | 7.40 MB | 7.39 MB | 1.00x | 1.78x | FULL 1, MOTION 5, TILED 54 |
| mixed | 288.47 KB | **247.82 KB** | **1.16x** | 54.47x | FULL 1, PIXEL 17, MOTION 7, TILED 35 |

Exact bytes: screen 2,174 · cut 355,334 · slide 382,503 · noise 7,744,550 · mixed 253,772.

`mixed` is the clip built specifically to need this, and it delivers: `TILED` wins 35 of 59
frames. Its four quadrants want four different answers — static UI chrome, a sliding block,
an unchanged region, and abrupt repaints — and until now one frame-level choice had to
compromise across all four.

`screen` gains 1.07x and, with it, **beats the reference implementation running every mode**:
2,174 bytes against 2,267. Its changes are a blinking cursor and text appearing one character
at a time, which is exactly the pattern where most of the frame wants "nothing happened" and
one small region wants a delta.

## Observation 1: winning the ranking is not the same as helping

`noise` picks `TILED` on **54 of 59 frames** and gains **0.1%** — 7,755,148 → 7,744,550.

That is not a contradiction. `TILED` is written whenever it prices smallest, and splitting a
uniformly noisy frame into 16 tiles each choosing `MOTION` is very slightly cheaper than one
frame-wide `MOTION`, because per-tile searches find marginally better vectors near tile
interiors. The gain is real and it is also nearly nothing.

The useful lesson is about reading mode counts: **a mode's win rate says how often it was
cheapest, not how much it was worth.** On `noise`, `TILED` is cheapest almost always and
worth almost nothing; on `mixed` it is cheapest 59% of the time and worth 1.16x. A report
showing only mode counts would rank these the same way round, which is why the size column
is the one that matters.

## Observation 2: grid 4 is right, but not everywhere

| clip | grid 1 | grid 2 | **grid 4** | grid 6 | best |
| --- | ---: | ---: | ---: | ---: | ---: |
| screen | 2.27 KB | 2.23 KB | **2.12 KB** | 2.20 KB | 4 |
| cut | 347.01 KB | 347.01 KB | 347.01 KB | 347.01 KB | — |
| slide | 373.54 KB | 373.47 KB | 373.54 KB | **360.33 KB** | 6 |
| noise | 7.40 MB | 7.39 MB | **7.39 MB** | 7.40 MB | 4 |
| mixed | 288.47 KB | 265.83 KB | **247.82 KB** | 259.56 KB | 4 |

Grid 1 is the control — a single tile, so `TILED` can never beat the frame-level choice and
the column matches M4 exactly, as it must.

Four wins on three clips and the curve turns over after it: `screen` and `mixed` both get
*worse* at 6, because per-tile overhead (a mode byte and a varint length each) starts
outrunning the benefit as tiles shrink. `mixed` is the clearest case — 4 is a 1.16x win, 6
gives back a third of it.

`slide` is the exception and an interesting one: grid 6 finds 3.5% that grid 4 cannot, because
finer tiles let the moving box's edge fall on a tile boundary instead of inside a tile. That
is the same edge-alignment problem the block-size sweep exposed at M4, one level up — and the
same answer applies: a fixed subdivision cannot suit all content, which is what variable-size
partitions are for.

Raising the grid is always *safe* for size, because `TILED` is only written when it actually
prices smallest. It is not free in time: cost grows linearly in the tile count.

## Observation 3: tiles are independent, and that costs something

Each tile's mode sees only its own slice of the current and reference frames, so a `MOTION`
tile can only find matches **within its own region**. An object crossing a tile boundary is
predicted worse than it would be frame-wide.

The cost is real but bounded: only blocks within `search_range` of a boundary are affected.
It is visible in the grid sweep as the turnover after 4 — part of what makes grid 6 worse on
`screen` and `mixed` is boundary loss, not just per-tile overhead.

Fixing it would mean giving every mode an origin offset into a shared reference frame, so a
tile could search outside itself. That is the right trade to revisit if boundary loss ever
dominates, and recording it now is cheaper than rediscovering it later.

## Observation 4: encode time is now the headline problem

| clip | M3 | M4 | M5 |
| --- | ---: | ---: | ---: |
| cut | <0.1s | 1.1s | 2.2s |
| screen | <0.1s | 5.2s | 6.6s |
| mixed | <0.1s | 9.1s | 16.0s |
| slide | <0.1s | 13.9s | 22.6s |
| noise | <0.1s | 19.5s | 38.1s |

`noise` has gone from under a tenth of a second at M3 to 38 seconds. The structure is
multiplicative: `trial` prices every frame-level mode, `TILED` is one of them, and pricing
`TILED` prices four modes on each of 16 tiles — including a full motion search per tile. So a
single frame can run 17 motion searches, and `cut` pays for all of them while choosing
`TILED` zero times.

This is no longer a nuisance; at 320x240 it is 38 seconds for 60 frames, and the search cost
scales with frame area. Both remaining speed milestones now have a measured case to argue
from: **M6**'s learned predictor avoids pricing modes that will lose, and **M9**'s
coarse-to-fine search makes each search cheaper.

## Where the remaining gap is

| clip | M5 (here) | reference, all modes | gap | what closes it |
| --- | ---: | ---: | ---: | --- |
| screen | **2,174** | 2,267 | **0.96x — ahead** | — |
| cut | 355,334 | 231,630 | 1.53x | M8 intra prediction |
| slide | 382,503 | 380,955 | 1.00x | — (grid 6 gives 360,338, also ahead) |
| noise | 7,744,550 | 7,223,174 | 1.07x | M8 intra prediction |
| mixed | 253,772 | 211,974 | 1.20x | M8 intra prediction |

**Two clips now match or beat the full reference**, and every remaining gap is intra
prediction — the one piece of the reference's design this rebuild has not yet built. M1
priced it at 249.8 KB across the five clips when `FULL` was chosen rarely; `cut` still spends
120 KB on a single `FULL` frame that intra prediction compresses to 71 bytes.

That makes M8 the largest remaining size lever, and M6/M9 the largest remaining time levers.
The mode set itself is essentially complete.
