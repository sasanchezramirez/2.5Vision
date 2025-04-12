from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ImageMetadata(BaseModel):
    latitude: Optional[float] = Field(default=None, description="Latitud de la imagen")
    longitude: Optional[float] = Field(default=None, description="Longitud de la imagen")
    datetime_taken: datetime
    visibility_score: int
    weather_tags: Optional[str] = None
    uploader_username: Optional[str] = None
    image_url: Optional[str] = None
    image_name: Optional[str] = None