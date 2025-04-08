from app.domain.model.image_metadata import ImageMetadata
from app.infrastructure.driven_adapter.persistence.entity.image_metadata_entity import ImageMetadataEntity
class ImageMapper:
    @staticmethod
    def map_image_metadata_to_entity(image_metadata: ImageMetadata) -> ImageMetadataEntity:
        return ImageMetadataEntity(
            latitude=image_metadata.latitude,
            longitude=image_metadata.longitude,
            datetime_taken=image_metadata.datetime_taken,
            visibility_score=image_metadata.visibility_score,
            weather_tags=image_metadata.weather_tags,
            uploader_username=image_metadata.uploader_username,
            image_url=image_metadata.image_url
        )       