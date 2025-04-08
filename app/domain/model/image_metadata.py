from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ImageMetadata(BaseModel):
    location: str
    datetime_taken: datetime
    visibility_score: int
    weather_tags: Optional[str] = None
    uploader_username: Optional[str] = None
    image_url: Optional[str] = None