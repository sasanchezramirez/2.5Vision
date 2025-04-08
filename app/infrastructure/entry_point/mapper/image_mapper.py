from datetime import datetime
from typing import Optional

from app.domain.model.image_metadata import ImageMetadata

class ImageMapper:
    @staticmethod
    def map_upload_image_request_to_image_metadata(latitude: float, longitude: float, datetime_taken: datetime, visibility_score: int, weather_tags: Optional[str] = None, uploader_username: Optional[str] = None) -> ImageMetadata:
        return ImageMetadata(
            datetime_taken=datetime_taken,
            visibility_score=visibility_score,
            weather_tags=weather_tags,
            uploader_username=uploader_username
        )