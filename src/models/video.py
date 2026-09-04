"""VideoInfo from ffprobe: width, height, fps as a rational, optional frame-count hint."""

from pydantic import BaseModel, ConfigDict


class VideoInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    fps_num: int
    fps_den: int
    num_frames: int | None = None
