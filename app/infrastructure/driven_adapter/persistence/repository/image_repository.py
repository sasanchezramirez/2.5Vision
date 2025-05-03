import logging
from typing import Final

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.domain.model.image_metadata import ImageMetadata
from app.infrastructure.driven_adapter.persistence.mapper.image_mapper import ImageMapper
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
from app.infrastructure.driven_adapter.persistence.entity.image_metadata_entity import ImageMetadataEntity

logger: Final[logging.Logger] = logging.getLogger("Image Repository")

class ImageRepository:
    """
    Implementación del repositorio de imágenes.
    
    Esta clase implementa las operaciones de persistencia para imágenes
    utilizando SQLAlchemy como ORM.
    """
    def __init__(self, session: Session):
        self.session: Final[Session] = session
    
    def create_image_metadata(self, image_metadata_entity: ImageMetadataEntity) -> ImageMetadata:
        """
        Crea un nuevo metadato de imagen en la base de datos.

        Args:
            image_metadata: Metadato de imagen a crear

        Returns:
            ImageMetadata: Metadato de imagen creado
        """
        logger.info(f"Creando metadato de imagen: {image_metadata_entity}")
        try:
            self.session.add(image_metadata_entity)
            self.session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Error al crear metadato de imagen: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)
        except Exception as e:
            logger.error(f"Error no manejado al crear metadato de imagen: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG01)
        
        
        