"""VideoInfo from ffprobe: width, height, fps as a rational, optional frame-count hint."""

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

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den
