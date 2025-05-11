from app.domain.model.image_metadata import ImageMetadata
from app.infrastructure.driven_adapter.persistence.entity.image_metadata_entity import ImageMetadataEntity

class ImageMapper:
    @staticmethod
    def map_image_metadata_to_entity(image_metadata: ImageMetadata) -> ImageMetadataEntity:
        return ImageMetadataEntity(
            latitude=image_metadata.latitude if image_metadata.latitude is not None else None,
            longitude=image_metadata.longitude if image_metadata.longitude is not None else None,
            datetime_taken=image_metadata.datetime_taken,
            visibility_score=image_metadata.visibility_score,
            weather_tags=image_metadata.weather_tags if image_metadata.weather_tags is not None else "",
            uploader_username=image_metadata.uploader_username if image_metadata.uploader_username is not None else "",
            image_url=image_metadata.image_url if image_metadata.image_url is not None else "",
            image_name=image_metadata.image_name if image_metadata.image_name is not None else ""
        )
    
    @staticmethod
    def map_entity_to_image_metadata(entity: ImageMetadataEntity) -> ImageMetadata:
        """
        Convierte una entidad ImageMetadataEntity a un objeto de dominio ImageMetadata.
        
        Args:
            entity: Entidad de base de datos a convertir
            
        Returns:
            ImageMetadata: Objeto de dominio convertido
        """
        return ImageMetadata(
            latitude=entity.latitude,
            longitude=entity.longitude,
            datetime_taken=entity.datetime_taken,
            visibility_score=entity.visibility_score,
            weather_tags=entity.weather_tags if entity.weather_tags is not None else "",
            uploader_username=entity.uploader_username if entity.uploader_username is not None else "",
            image_url=entity.image_url if entity.image_url is not None else "",
            image_name=entity.image_name if entity.image_name is not None else ""
        )       