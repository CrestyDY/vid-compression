"""Compressed video header"""

from pydantic import BaseModel, ConfigDict, Field

class Header(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_identifier: bytes = b"myv"
    pix_fmt_id: int = 0             # rgb24 for now
    width: int
    height: int
    fps_num: int
    fps_den: int
    fps_count: int = Field(default=0, ge=0)
