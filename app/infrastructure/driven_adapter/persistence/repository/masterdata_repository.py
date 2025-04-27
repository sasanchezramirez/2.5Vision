import logging
from typing import Final, List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, desc
from app.infrastructure.driven_adapter.persistence.entity.image_metadata_entity import ImageMetadataEntity
from app.domain.model.util.response_codes import ResponseCodeEnum
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.user import User
from app.infrastructure.driven_adapter.persistence.entity.user_entity import UserEntity
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
        
    def get_top_users_by_images_uploaded(self) -> List[Tuple[UserEntity, int]]:
        """
        Obtiene los usuarios con más imágenes subidas.

        Returns:
            List[Tuple[UserEntity, int]]: Lista de tuplas con usuario y cantidad de imágenes subidas
        """
        try:
            query_result = (
                self.session.query(
                    UserEntity, 
                    func.count(ImageMetadataEntity.id).label('total_images')
                )
                .join(
                    ImageMetadataEntity,
                    UserEntity.username == ImageMetadataEntity.uploader_username
                )
                .group_by(UserEntity.id)
                .order_by(desc('total_images'))
                .limit(3)
                .all()
            )
            
            return query_result
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener los usuarios con más imágenes subidas: {e}")
            raise CustomException(ResponseCodeEnum.KOG02)
    
    def get_total_images_uploaded_by_user(self, username: str) -> int:
        """
        Obtiene el total de imágenes subidas por un usuario.

        Args:
            username (str): Nombre de usuario del usuario a buscar.

        Returns:
            int: Total de imágenes subidas
        """ 
        try:
            return self.session.query(func.count(ImageMetadataEntity.id)).filter(ImageMetadataEntity.uploader_username == username).scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener el total de imágenes subidas por usuario: {e}")
            raise CustomException(ResponseCodeEnum.KOG02)
