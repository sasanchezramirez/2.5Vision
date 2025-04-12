from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ImageUploadResponse(BaseModel):
    image_url: str
    image_name: str
    image_type: str
    image_size: int 

class ImageMetadataResponse(BaseModel):
    latitude: float = Field(default=0.0, description="Latitud de la imagen")
    longitude: float = Field(default=0.0, description="Longitud de la imagen")
    datetime_taken: datetime
    visibility_score: int
    weather_tags: Optional[str] = None
    uploader_username: Optional[str] = None
    image_url: str