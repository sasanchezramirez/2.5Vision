import logging
from typing import Final

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func
from app.infrastructure.driven_adapter.persistence.entity.image_metadata_entity import ImageMetadataEntity
from app.domain.model.util.response_codes import ResponseCodeEnum
from app.domain.model.util.custom_exceptions import CustomException

logger: Final[logging.Logger] = logging.getLogger("Masterdata Repository")

class MasterdataRepository:
    def __init__(self, session: Session):
        self.session = session
        
    def get_total_images_uploaded(self) -> int:
        """
        Obtiene el total de imágenes subidas por los usuarios.

        Returns:
            int: Total de imágenes subidas
        """
        try:
            return self.session.query(func.count(ImageMetadataEntity.id)).scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener el total de imágenes subidas: {e}")
            raise CustomException(ResponseCodeEnum.KOG02)
