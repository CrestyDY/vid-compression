# A real entropy coder — mostly a negative result

M8's comparison against FFV1 made this the next thing to build: FFV1 is intra-only, with no
motion compensation at all, and it still beat this codec **1.38x on real footage** after intra
prediction landed. The remaining difference had to be the entropy stage — FFV1 adapts a
probability model per symbol from its neighbours, while zstd is a general-purpose LZ matcher
with no idea it is looking at image residuals.

So: a context-modelled binary arithmetic coder, 2,048 adaptive contexts, verified by
round-tripping thousands of random bit sequences at random probabilities.

**It does not beat zstd.** Measured across 1.94 MB of this codec's real payloads, it is
**0.98x** — a 2% gain, nowhere near the 1.38x the FFV1 comparison implied was available. The
interesting part is *why*, and what came out of it instead.

## The measurement

Real pre-entropy payloads extracted from actual encodes, compressed both ways:

| clip | payload | zstd | range | range/zstd |
| --- | ---: | ---: | ---: | ---: |
| screen | 280.82 KB | **1.89 KB** | 4.03 KB | **2.13x worse** |
| cut | 450.08 KB | **225.33 KB** | 228.31 KB | 1.01x worse |
| slide | 391.67 KB | 249.92 KB | **243.92 KB** | 0.98x |
| noise | 450.00 KB | 234.59 KB | **222.33 KB** | **0.95x** |
| mixed | 411.52 KB | 138.30 KB | **137.45 KB** | 0.99x |
| **total** | **1.94 MB** | **850.03 KB** | **836.04 KB** | **0.98x** |

Broken down by which mode produced the payload, the pattern is much sharper:

| mode | payload | zstd | range | range/zstd |
| --- | ---: | ---: | ---: | ---: |
| `FULL` | 1.54 MB | 755.86 KB | **736.49 KB** | **0.97x** |
| `PIXEL_DELTA` | 228 B | 741 B | **228 B** | **0.31x** |
| `BLOCK_DELTA` | 24.98 KB | **974 B** | 2.89 KB | **3.03x worse** |
| `MOTION` | 167.44 KB | **66.98 KB** | 69.21 KB | 1.03x worse |
| `TILED` | 216.44 KB | **25.51 KB** | 27.22 KB | 1.07x worse |

## Why: LZ and context modelling are good at different things

One `BLOCK_DELTA` payload, inspected directly:

```
screen BLOCK_DELTA 3.02 KB: 90.5% zeros, longest zero run 705 bytes
  zstd   84 B
  range 257 B
```

That is the whole explanation. A 705-byte run of zeros is **one LZ match** — zstd spends a
few bytes on it. An arithmetic coder has to code all 705 symbols individually, and even at
P(0) ≈ 0.999 that is a nonzero cost each, plus the model's adaptation lag at the start of
every run.

**LZ wins on exact repeats; context modelling wins on skewed but non-repeating
distributions.** These payloads contain both: dense intra-predicted pixel fields where the
range coder wins (`FULL`, 0.97x), and vast runs of zeros where zstd wins by 3x
(`BLOCK_DELTA`). zstd was already capturing the repeat structure, so the headroom the FFV1
comparison suggested was not where the comparison implied.

This is also why `screen` is the worst case at 2.13x: its payloads are small, sparse and
run-heavy, which is zstd's best case and the arithmetic coder's worst.

## What came out of it: price both, keep the smaller

Neither codec dominates, which is exactly the situation this codec already has an answer for.
So the entropy stage gained an `auto` setting that compresses each payload both ways and keeps
the smaller, spending one byte to record which won — **the same argument the codec makes for
modes, one layer down.**

That gives a guarantee worth stating: `auto` cannot lose to zstd by more than one byte per
frame, and it collects the range coder's wins wherever they exist. A test pins it.

End to end, lossless throughout:

| clip | zstd | `auto` | gain | encode time |
| --- | ---: | ---: | ---: | --- |
| screen | 2,268 | **2,024** | **1.12x** | 8.3s → 158.1s |
| cut | 231,630 | **231,234** | 1.002x | 4.1s → 234.1s |
| slide | 382,552 | **374,089** | **1.02x** | 22.6s → 373.4s |
| mixed | 205,814 | *not measured* | — | — |
| noise | 7,223,174 | *not measured* | — | — |

`screen`'s 1.12x is larger than the payload table suggests it should be, and that is the
point: per-payload choice beats either codec used alone. On the clip where the range coder
was 2.13x *worse* overall, selecting per payload still found 10.8%. It also puts `screen`
back ahead of the reference implementation, at 2,024 against 2,267.

> **Two rows are missing on purpose.** `mixed` and `noise` were not measured end to end,
> because at 130–180 KB/s the arithmetic coder needs tens of minutes per clip and `noise` is
> the largest. The payload-level table above covers them (`noise` 0.95x, `mixed` 0.99x), and
> `noise` is the clip where the coder does best, so `auto` should gain a little more there
> than on `slide`. Reproduce with:
>
> ```
> uv run vidc encode bench/clips/noise.mp4 out.myv --entropy auto
> ```
>
> Nothing depends on those two numbers — the conclusion below rests on the payload-level
> measurement and the three clips that were measured.

## Honest accounting

**The FFV1 gap is not closed.** It was 1.38x after M8 and this work does not meaningfully
move it, because the gain is 2% where I expected tens of percent. The diagnosis in M8's
write-up — "what remains is a context-modelled range coder" — was **half right at best**: a
range coder alone is not what FFV1's advantage consists of.

What FFV1 actually has that this does not:

1. **2D gradient contexts.** Its context comes from the pixels above, left and above-left —
   genuinely two-dimensional. This coder's context is the *previous byte in scan order*,
   which is a far weaker predictor and is what caps the `FULL` gain at 3%.
2. **An explicit run mode.** FFV1 detects runs and codes their length, rather than coding
   each symbol. That is precisely the structure zstd exploits and this coder cannot.

So the next honest step is not "a better arithmetic coder" but **those two specific features**,
and the `BLOCK_DELTA` row above says the run mode is worth more than the contexts.

**On speed.** Arithmetic coding is inherently sequential — each symbol's interval depends on
the last — so unlike the rest of this codec it cannot be vectorised. In Python it runs at
roughly 130–180 KB/s, making `auto` 19–57x slower than zstd. It is correct, it measures the
idea, and `zstd` remains the default. The CLI prints a warning when a non-default codec is
selected rather than letting anyone wait in the dark.

**What it cannot do that zstd can.** An arithmetic stream has no internal redundancy, so a
corrupted-but-correct-length payload decodes to wrong pixels silently instead of raising.
Container-level truncation is still caught (the frame record stores `payload_len`), and
`verify` catches the rest. A test documents this rather than pretending otherwise.

## Where this leaves the project

The codec's own thesis survived contact with a layer it was not designed for: faced with two
entropy stages that win on different structure, the right answer turned out to be the same
one it already uses for modes. That is a better outcome than the 1.38x would have been.

The remaining size levers, now priced rather than guessed:

- **A run mode in the arithmetic coder** — worth up to 3x on `BLOCK_DELTA`-shaped payloads.
- **2D gradient contexts** — the `FULL` gain is 3% with order-1; FFV1 gets substantially more.
- **A source-native pixel format** — still the largest single item, worth roughly half the
  data on real `yuv420p` footage. See [ROADMAP.md](ROADMAP.md).
