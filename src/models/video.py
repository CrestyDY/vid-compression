"""VideoInfo from ffprobe: dimensions, frame rate, pixel format, frame-count hint."""

from pydantic import BaseModel, ConfigDict


class VideoInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    #: Frame rate as a rational, never a float, so 30000/1001 survives exactly.
    fps_num: int
    fps_den: int
    #: A hint from the container; often absent and sometimes wrong. Never a loop bound.
    num_frames: int | None = None
    #: The source's own pix_fmt, verbatim from ffprobe. What the encoder follows when it
    #: picks an internal format, so the codec stores the samples the file holds rather
    #: than ffmpeg's conversion of them. None when ffprobe did not say.
    pix_fmt: str | None = None

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den
