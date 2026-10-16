# Which lever is actually worth pulling

After the entropy coder came back as a 0.98x negative result, [that document](entropy-results.md)
named two things FFV1 has and this codec does not — **2D gradient contexts** and **an
explicit run mode** — and claimed the run mode was worth more, "up to 3x on
`BLOCK_DELTA`-shaped payloads".

That claim was wrong, and so was its diagnosis. This document is the measurement that
replaces it. All three candidates were priced before any of them was built, by simulating
the adaptive models' cost rather than implementing them: run the same binary models the real
coder uses and sum `-log2(p)`. That is the coder's output size to within a byte or two, and
it costs one pass over a payload instead of a codec.

## Where the arithmetic coder's bits actually go

Instrumenting `encode_bytes` to attribute cost per input byte, on real payloads taken from a
`--no-entropy` encode:

| clip | mode | n | payload | zstd | range | zeros | in runs >= 8 | bits spent on zero bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| screen | FULL | 1 | 230,401 | 156 | 382 | 100.0% | 100.0% | **59.8%** |
| screen | PIXEL_DELTA | 59 | 11,352 | 3,767 | 6,555 | 28.8% | 0.0% | 18.4% |
| mixed | FULL | 1 | 230,401 | 115,764 | 113,418 | 50.0% | 49.6% | 1.5% |
| mixed | TILED | 40 | 758,541 | 104,175 | 109,453 | 84.9% | 83.7% | 7.2% |

`screen`'s first frame is the degenerate case: 230,401 bytes that are *entirely* zero after
intra prediction. The coder spends 229 of its 382 bytes on them. That is the saturation
floor — `_PROB_MAX` is 65535 and the adaptation fixed point at `_RATE = 5` is 65504, so a
fully-saturated zero byte still costs 8 x log2(65536/65504) = 0.0056 bits, and 230,401 of
those add up. An explicit run mode would make this payload cost about 10 bytes instead of
382, a 38x win on this one payload.

## Why that 38x does not generalise

It only appears when a run is *uninterrupted*. On `mixed`'s `TILED` payloads — 84.9% zeros,
83.7% of all bytes inside runs of 8 or more, longest run 576 — the coder spends 7,880 bytes
on 643,000 zero bytes. That is **0.098 bits per zero byte**, 17x above the saturation floor.

The reason is the context, and it is not a defect. A zero byte inside a run is coded in
bucket 0 (previous byte near zero), and bucket 0's models are shared between "this run
continues" and "this run ends here". Run-end events keep knocking the probability back off
saturation, so the equilibrium sits near the *actual* run-end rate. Which is correct: with
runs averaging about 100 bytes, log2(100) + 1.44 is roughly 8 bits to describe one run, or
0.08 bits per byte in it.

**0.098 measured against a 0.08 information-theoretic floor means the existing coder is
already near-optimal on runs.** An explicit exp-Golomb run length for a 92-zero run costs
about 12 bits where the current coder spends 9. The run mode is not a 3x win waiting to be
collected; on everything except a degenerate all-zero payload it is break-even or worse.

And the degenerate case is already covered: zstd codes that same all-zero payload in 156
bytes, and `--entropy auto` picks zstd for it. **The run mode's entire measured benefit is
something `auto` already captures.** Not built.

## 2D gradient contexts: real, but small

Four context models simulated on the same payloads. `order1` is what ships (bucket of the
previous byte in scan order); the others add the byte one row above, which is available
because the payload is a filtered plane of known stride.

On `mixed` frame 0 (synthetic, noise-heavy, 230,401 bytes):

| model | size | vs order1 | vs zstd |
|---|---:|---:|---:|
| order1 | 113,418 | 1.000x | 1.021x |
| 2d-8x8 | 112,009 | 1.013x | 1.034x |
| 2d-8x3 | 111,683 | **1.016x** | 1.037x |
| grad-8x8 | 112,317 | 1.010x | 1.031x |

On a 640x360 crop of real 1080p footage, frame 0 (691,201 bytes):

| model | size | vs order1 | vs zstd |
|---|---:|---:|---:|
| order1 | 272,566 | 1.000x | **0.523x** |
| 2d-8x8 | 260,181 | **1.048x** | 0.548x |
| 2d-8x3 | 262,100 | 1.040x | 0.544x |
| grad-8x8 | 261,656 | 1.042x | 0.545x |

So 2D context is worth 1.6% on synthetic content and 4.8% on real content. Real, repeatable,
and far too small to be the FFV1 gap — which was 1.38x.

The second table also says something louder than it says about contexts: **on real footage
the arithmetic coder is 1.9x worse than zstd**, not the 1.03x it manages on synthetic clips.
No context tweak closes that. Something about real footage suits an LZ matcher enormously,
and finding out what it was is what produced the next section.

## The answer: we are compressing data the source never contained

Our pipeline asks ffmpeg for `rgb24`. `samples/real12.mp4` is `yuv420p`. So ffmpeg
*invents* three quarters of the chroma samples by upsampling, and every stage of this codec
then spends bytes losslessly preserving the invention. It also explains the zstd anomaly:
chroma upsampling makes neighbouring pixels exactly equal in 2x2 blocks, which is an LZ
matcher's ideal input and nothing special to a context model.

Measured on `samples/real12.mp4`, 1920x1080, intra prediction plus zstd, no temporal
prediction at all — the same frames, read two ways:

| frame | rgb24 raw | rgb24 coded | yuv420p raw | yuv420p coded (Y + U + V) |
|---|---:|---:|---:|---:|
| 0 | 6,220,800 | 761,844 | 3,110,400 | 555,089 (478,585 + 42,196 + 34,308) |
| 1 | 6,220,800 | 845,005 | 3,110,400 | 583,342 (491,261 + 49,874 + 42,207) |
| 2 | 6,220,800 | 837,771 | 3,110,400 | 585,193 (494,164 + 48,909 + 42,120) |
| **total** | 18,662,400 | **2,444,620** | 9,331,200 | **1,723,624** |

**1.42x smaller, before any of this codec's own modes are involved.** Against a measured
FFV1 deficit of 1.38x, that is not merely the largest remaining item — it is very nearly the
whole of it. FFV1 was never winning on entropy coding. It was winning because it codes the
samples the file actually holds, and we were coding ffmpeg's interpolation of them.

## Ranking, corrected

| candidate | predicted before | measured | built |
|---|---|---|---|
| explicit run mode | "worth more", up to 3x | 38x on one degenerate payload, break-even elsewhere, already covered by `auto` | no |
| 2D gradient contexts | the other half of FFV1 | 1.6% synthetic, 4.8% real | not yet |
| source-native pixel format | "half the data" | **1.42x on real footage, intra-only** | M12 |

The reproduction scripts for every table here are throwaway instrumentation, not shipped
code; the payload dumps they read come from `vidc encode <clip> <out> --no-entropy`, whose
container stores pre-entropy payloads verbatim.
