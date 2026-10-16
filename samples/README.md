# samples/

Drop real `.mp4` files here to measure the codec against actual footage.

Everything in this folder except this README is gitignored, so put whatever you like in it.

```bash
uv run vidc encode samples/yourclip.mp4 out.myv
uv run vidc verify samples/yourclip.mp4 out.myv
```

## `real_yuv.mp4`, for the real-footage tests

A handful of tests in `tests/test_planar.py` measure the source-native pixel format against
real content, because the synthetic benchmark clips cannot: they are `gbrp`, so there is no
chroma subsampling for the milestone to exploit. Those tests **skip** unless
`samples/real_yuv.mp4` exists.

Make one from any `yuv420p` footage you have. A small crop rather than a scale, so the
statistics stay native:

```bash
ffmpeg -i your-1080p-clip.mp4 -vf "crop=320:240:800:400" -frames:v 4 \
       -c:v libx264 -qp 0 -preset veryslow -pix_fmt yuv420p \
       samples/real_yuv.mp4
```

`libx264 -qp 0` is lossless here precisely because the input is already `yuv420p` and no
colour conversion happens — which is the same reason it is the *wrong* choice for an RGB
clip, where `libx264rgb -qp 0` is needed instead.

The tests assert a ratio, not a byte count, so any real `yuv420p` footage works. They will
fail on synthetic flat-colour content, and should: that content has nothing to measure.

## What to expect

Check the pixel format line in the size report first. If it says `yuv420p` the codec is
reading the file's own samples; if it says `rgb24` on footage you expected to be YUV, the
encoder will have printed a note saying why.

Real footage is still much larger through this codec than through H.264 — the source MP4 is
lossy and this codec is not, so it is a category difference rather than a defeat. The
comparisons that mean something are against other *lossless* codecs, and as of M12 those are
FFV1 at 1.14x ahead and x265 lossless at 1.62x ahead on 1080p
([results](../docs/m12-results.md)).

The interesting number in the size report is `changed pixels/frame`. On live-action footage
it tends to be most of the frame, because sensor noise and dithering mean almost nothing is
ever *exactly* unchanged — which is why `PIXEL_DELTA` alone could produce a file larger than
the raw frames it describes, and why M2's per-frame mode selection exists. Screen recordings
behave completely differently. That contrast is the whole lesson.
