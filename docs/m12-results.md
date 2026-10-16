# M12 — the source-native pixel format

**Measured: 1.55x smaller on real 1080p footage, 2.25x faster, and the FFV1 deficit cut from
1.77x to 1.14x.** The five synthetic clips are byte-for-byte unchanged.

This is the largest single improvement in the rebuild, and it is not a new mode, a better
predictor or a better entropy coder. It is the codec stopping compressing data that was
never in the file.

## The problem

Every stage of this codec dealt in `rgb24`. For the synthetic benchmark clips that is
correct — they are `gbrp`, true lossless RGB from `libx264rgb -qp 0`, and `gbrp → rgb24` is
a plane reorder that touches no sample value.

Real footage is not RGB. `samples/real12.mp4` is `yuv420p`: one luma sample per pixel and
one chroma pair per 2x2 block. Asking ffmpeg for `rgb24` makes it **interpolate** the three
missing chroma samples in every block, and the codec then spent bytes losslessly preserving
ffmpeg's interpolation — detail the source file never contained.

The reference implementation's roadmap frames this as "switch to a YUV420 pipeline". That is
wrong as stated, and measurably so: forcing a synthetic clip through `yuv420p` corrupts
**94.2% of its bytes, max error 250**. A global switch would have broken losslessness on all
five benchmark clips.

The fix is to make the internal format *follow the source* and never convert.

## How it was priced

This item sat at the bottom of the "beyond M10" list for several milestones because its
value was estimated ("half the data through every stage") rather than measured. It was
promoted after [next-lever.md](next-lever.md) priced all three remaining candidates:

| candidate | estimated before | measured |
| --- | --- | --- |
| explicit run mode | "worth more", up to 3x | break-even; its one win already belongs to `auto` |
| 2D gradient contexts | the other half of FFV1 | 1.6% synthetic, 4.8% real |
| **source-native pixel format** | "half the data" | **1.42x on real footage, intra-only** |

The 1.42x was intra prediction plus zstd with no temporal prediction at all, against a
measured FFV1 deficit of 1.38x. That made the ordering obvious: the FFV1 gap was very nearly
*entirely* this, and had nothing to do with entropy coding.

## The design

**A frame is a tuple of planes.** `RGB24` is a one-plane format whose single plane carries
three channels; `yuv420p` is three planes, the last two at a quarter the area. The one-plane
case is a tuple of length one rather than a special case, which is what keeps every stage
above `src/pixfmt.py` from branching on the format.

**Each plane chooses its own mode.** `FrameType.PLANES` wraps one chosen mode per plane:

```
count   : u8        number of planes
then per plane, in the format's own order:
    mode    : u8
    length  : uvarint
    payload : length bytes
```

This is the argument `TILED` makes about regions of a frame, one level further out. Luma
carries the detail and the motion; chroma is a quarter the size, smoother, and frequently
unchanged between frames where luma is not. It is exercised. Four frames of real footage, with each plane's chosen mode and its
pre-entropy payload size:

| frame | Y | U | V |
| --- | --- | --- | --- |
| 0 | FULL 76,801 | FULL 19,201 | FULL 19,201 |
| 1 | TILED 78,316 | **MOTION** 16,694 | TILED 19,245 |
| 2 | MOTION 78,906 | MOTION 14,902 | MOTION 14,632 |
| 3 | TILED 77,881 | TILED 18,531 | TILED 19,252 |

Frame 1 is the case the design exists for: luma wants `TILED`, chroma-U wants `MOTION`.

**But per-plane mode freedom is worth almost nothing on its own.** On this clip, choosing
per plane beats the best single mode by **1.001x**. Splitting the planes is a structural
requirement — they have different shapes, so they *cannot* share one payload — and the
freedom that falls out of it is a rounding error. All of the 1.55x comes from not
interpolating. Saying so matters, because the per-plane machinery is the visible part of
this milestone and it is not the part that pays.

**Single-plane files are untouched.** An `rgb24` encode still stores a bare mode in the frame
record, so there was no format version bump: nothing about an `rgb24` payload changed, and a
planar file is distinguished by a pixel format id that older readers already reject with a
clear message. Bumping would have invalidated every existing V2 file for nothing.

**Full-range variants get their own ids.** `yuvj420p` differs from `yuv420p` only by a
colour-range tag, but asking ffmpeg for the wrong one of the two makes swscale apply a range
conversion, and a conversion rounds. A codec that rounds is not lossless, so the stored
format names the exact `pix_fmt` to ask for and never a compatible-looking neighbour.

## Results — real 1080p footage

`samples/real12.mp4`, 1920x1080, 12 frames, `yuv420p`, 388,287 bytes as H.264. Raw frames
are 71.19 MB as `rgb24` and 35.59 MB as `yuv420p`. Lossless throughout, `verify` confirmed.

| | size | encode time | vs previous build |
| --- | ---: | ---: | ---: |
| M11 build (`rgb24`) | 9,497,624 | 322.0s | — |
| **M12 (`yuv420p`, native)** | **6,114,014** | **143.3s** | **1.55x smaller, 2.25x faster** |

The speedup is not a bonus optimisation; it is the same fact stated twice. Two thirds of the
bytes means two thirds of the motion searches, residuals and zstd passes.

Against the reference codecs on the same file:

| codec | size | vs M12 |
| --- | ---: | ---: |
| x265 lossless | 3,778,935 | 1.62x ahead |
| FFV1 | 5,366,968 | **1.14x ahead** (was 1.77x) |
| **this codec, M12** | **6,114,014** | — |
| this codec, M11 | 9,497,624 | 1.55x behind |

**FFV1's lead fell from 1.77x to 1.14x.** It is an intra-only codec with no motion
compensation whatsoever, and it had been beating a codec with four temporal modes purely
because it read the file's own samples while this one read an interpolation of them.

## Results — a smaller real clip, where the whole matrix is affordable

`samples/real_yuv.mp4`, 320x240, 4 frames, cropped from the same 1080p source so the
statistics are native rather than scaled. Small enough that the Python arithmetic coder is
usable, which turns out to matter a great deal. Every row lossless and `verify`-confirmed.

| configuration | size | vs the M11 build |
| --- | ---: | ---: |
| `rgb24` + `range` | 340,967 | 0.59x — *worse* |
| `rgb24` + zstd (the M11 build) | 200,548 | — |
| `yuv420p` + zstd | 147,543 | 1.36x |
| `yuv420p` + `auto` | 124,785 | 1.61x |
| **`yuv420p` + `range`** | **124,782** | **1.61x** |

For reference on the same clip: FFV1 126,186, x265 lossless 119,148.

### The entropy coder was never the problem; its input was

The two `range` rows are the most interesting result in this milestone, and they were not
expected.

[entropy-results.md](entropy-results.md) concluded that the context-modelled arithmetic
coder was a **0.98x negative result**, and on real footage it was far worse than that — 1.70x
*worse* than zstd on this very clip. The reason is now obvious in hindsight: chroma
upsampling makes neighbouring pixels **exactly equal** in 2x2 blocks. That is an LZ matcher's
ideal input and nothing whatsoever to a context model. zstd was winning on structure ffmpeg
had manufactured.

On the source's own planes that structure is gone, and the ranking inverts: **the range coder
beats zstd by 1.18x** (124,782 against 147,543) on the one kind of content that actually
matters. Combined with the format change it is **1.61x** against the M11 build, which is
enough to put this codec slightly *ahead of FFV1* on this clip — 124,782 against 126,186,
with `--entropy range`.

So the entropy coder was not a bad idea that happened to measure badly. It was a good idea
being fed the wrong bytes, and the measurement that condemned it was measuring an artefact.
It stays off by default only because it runs at 130–180 KB/s in Python.

## The synthetic clips: deliberately unchanged

All five are `gbrp`, so `for_source` maps them to `RGB24` and nothing about their encode
path changed. Confirmed byte-identical to M8 **and** re-verified lossless:

| clip | M8 | M12 | lossless |
| --- | ---: | ---: | --- |
| screen | 2,268 | 2,268 | yes |
| cut | 231,630 | 231,630 | yes |
| slide | 382,552 | 382,552 | yes |
| mixed | 205,814 | 205,814 | yes |
| noise | 7,223,174 | 7,223,174 | yes |

Byte-identical output across an architecture change this wide is the useful result here: the
one-plane path is genuinely the same path, not a re-implementation that happens to agree.

## Formats supported

`rgb24`, `yuv420p`, `yuv422p`, `yuv444p`, `yuvj420p`, `yuvj422p`, `yuvj444p`, `gray`. Each
round-trips losslessly end to end. Sources that are a reordering of one of these — `gbrp`,
`nv12`, `yuyv422`, `pal8` — map to it silently, because the reorder costs no sample value.

Anything else falls back to `rgb24` **loudly**:

```
$ vidc encode ten-bit.mp4 out.myv
  note: source pixel format 'yuv420p10le' is not one this codec stores natively; coding
  as rgb24. The result is still lossless with respect to the rgb24 frames ffmpeg
  produces, but those are a conversion of the source, not the source itself.
```

That distinction is the whole milestone in one sentence, so it is worth printing rather than
burying. Alpha-bearing formats are deliberately *not* in the silent map: `rgb24` discards
the alpha channel, which is a real loss and the user should hear about it.

## What this cost

**A bug that only a planar source could expose.** `PIXEL_DELTA`'s record had a fixed 3-byte
delta field. On a one-channel chroma plane the deltas broadcast silently into it on the way
in — writing each value three times, wasting two bytes per pixel — and then refused to
broadcast back out on the way in reverse:

```
frame 3 (PLANES): non-broadcastable output operand with shape (90,1)
doesn't match the broadcast shape (90,3)
```

Encode-but-never-decode is the worst shape a format bug can have: nothing complains until
the file is needed. It was found by running *every* encode flag against a planar source
rather than only the default one — the default happened never to choose `PIXEL_DELTA` on real
footage, while `--selection fixed`, `--selection actual` and `--no-entropy` all do. The
record is now `4 + c` bytes, derived from the plane's shape, which the decoder already has;
an RGB record is still exactly 7 bytes. The other four modes were already using `shape[2]`
correctly.

**A limit worth stating.** On a tiny clip with flat chroma the per-plane overhead makes the
native path *larger*: a `PLANES` frame costs 1 byte plus a mode and a length per plane, and
on the 32x24 test fixtures that is most of the file. A test pins this rather than leaving it
to be rediscovered. The 1.5x is a claim about real footage and is stated as one.

**One deferred minor, collected.** `CodecStats.raw_frame_bytes` hardcoded a `* 3` that only
RGB24 ever justified. It was on the deferred list with a note that it would bite at the YUV
work, and it did.

## Where this leaves the project

The remaining measured gaps on real footage are now x265's 1.62x and FFV1's 1.14x, against
2.51x and 1.77x before this milestone. What is left is no longer a mystery:

1. **Make the arithmetic coder usable at 1080p.** It is worth 1.18x on native planes and
   cannot be run on real footage because it is pure Python. This is now the largest *known*
   win in the project and it requires no new ideas, only a faster inner loop.
2. **2D gradient contexts**, worth a further 4.8% — and worth more now, since the content
   reaching the coder is no longer dominated by interpolation artefacts. The 4.8% was
   measured on `rgb24` payloads; it should be re-measured on planes.
3. **Variable-size motion partitions**, still the largest untouched structural item.
4. **Sub-pixel motion compensation**, which [the roadmap](ROADMAP.md) correctly said to
   measure only *after* this milestone.

The run mode is not on that list, and neither is "a better entropy coder" in the abstract.
Both were priced and found wanting; what the priced measurement found instead was a pixel
format.
