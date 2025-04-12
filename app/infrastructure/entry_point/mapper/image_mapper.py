from datetime import datetime
from typing import Optional

from app.domain.model.image_metadata import ImageMetadata
from app.infrastructure.entry_point.dto.image_dto import ImageMetadataResponse

class ImageMapper:
    @staticmethod
    def map_upload_image_request_to_image_metadata(datetime_taken: datetime, visibility_score: int, weather_tags: Optional[str] = None, uploader_username: Optional[str] = None) -> ImageMetadata:
        return ImageMetadata(
            datetime_taken=datetime_taken,
            visibility_score=visibility_score,
            weather_tags=weather_tags,
            uploader_username=uploader_username
        )
    
    @staticmethod
    def map_upload_image_response_to_image_metadata(image_metadata: ImageMetadata) -> ImageMetadataResponse:
        return ImageMetadataResponse(
            latitude=image_metadata.latitude if image_metadata.latitude is not None else 0.0,
            longitude=image_metadata.longitude if image_metadata.longitude is not None else 0.0,
            datetime_taken=image_metadata.datetime_taken,
            visibility_score=image_metadata.visibility_score,
            weather_tags=image_metadata.weather_tags,
            uploader_username=image_metadata.uploader_username,
            image_url=image_metadata.image_url
        )