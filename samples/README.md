# samples/

Drop real `.mp4` files here to measure the codec against actual footage.

Everything in this folder except this README is gitignored, so put whatever you like in it.

```bash
uv run vidc encode samples/yourclip.mp4 out.myv
uv run vidc verify samples/yourclip.mp4 out.myv
```

A note on what to expect at M1: the `.myv` will be **much larger** than the source MP4,
often by 10–50x on real video. That is the correct result. H.264 spent decades getting good
at this; a coordinate-tagged list of changed pixels has not.

The interesting number in the size report is `changed pixels/frame`. On live-action footage
it tends to be most of the frame, because sensor noise and dithering mean almost nothing is
ever *exactly* unchanged — which is why `PIXEL_DELTA` can produce a file larger than the raw
frames it describes. Screen recordings behave completely differently. That contrast is the
whole lesson, and it is what M2 exists to exploit.
