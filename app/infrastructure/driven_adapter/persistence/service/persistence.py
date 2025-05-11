import logging
from typing import Optional, Final, List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import asyncio
from sqlalchemy import text

from app.infrastructure.driven_adapter.persistence.entity.user_entity import UserEntity
from app.infrastructure.driven_adapter.persistence.repository.user_repository import UserRepository
from app.domain.model.user import User
from app.domain.gateway.persistence_gateway import PersistenceGateway
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
import app.infrastructure.driven_adapter.persistence.mapper.user_mapper as mapper
from app.infrastructure.driven_adapter.persistence.mapper.image_mapper import ImageMapper
from app.domain.model.image_metadata import ImageMetadata
from app.infrastructure.driven_adapter.persistence.repository.image_repository import ImageRepository
from app.infrastructure.driven_adapter.persistence.repository.masterdata_repository import MasterdataRepository

logger: Final = logging.getLogger("Persistence")

class Persistence(PersistenceGateway):
    """
    Implementación del gateway de persistencia.
    
    Esta clase implementa las operaciones de persistencia utilizando
    SQLAlchemy como ORM.
    """
    def __init__(self, session: Session):
        self.session = session
        self.user_repository = UserRepository(session)
        self.image_repository = ImageRepository(session)
        self.masterdata_repository = MasterdataRepository(session)
        
    async def __ensure_session_health(self) -> None:
        """
        Verifica que la sesión esté en buen estado y la reinicia si es necesario. Es un health check
        """
        try:
            self.session.execute(text("SELECT 1")).scalar()
        except Exception as e:
            logger.warning(f"Detectado problema de sesión, reiniciando: {e}")
            self.session.close()
            await asyncio.sleep(0.5)

    async def create_user(self, user: User) -> User:
        """
        Crea un nuevo usuario en la base de datos.

        Args:
            user: Usuario a crear

        Returns:
            User: Usuario creado con ID asignado
        """
        await self.__ensure_session_health()
        logger.info("Inicia el flujo de creación de usuario en base de datos")
        try:
            result = await self.user_repository.create_user(user)
            return result
        except CustomException as e:
            raise e

    async def get_user_by_id(self, id: int) -> Optional[User]:
        """
        Obtiene un usuario por su ID.

        Args:
            id: ID del usuario a buscar

        Returns:
            Optional[User]: Usuario encontrado o None si no existe
        """
        await self.__ensure_session_health()
        logger.info(f"Inicia el flujo de obtención de usuario por ID en base de datos. ID: {id}")
        try:
            result = await self.user_repository.get_user_by_id(id)
            return result
        except CustomException as e:
            raise e

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Obtiene un usuario por su nombre de usuario.

        Args:
            username: Nombre de usuario del usuario a buscar

        Returns:
            Optional[User]: Usuario encontrado o None si no existe
        """
        await self.__ensure_session_health()
        logger.info(f"Inicia el flujo de obtención de usuario por nombre de usuario en base de datos. Username: {username}")
        try:
            result = await self.user_repository.get_user_by_username(username)
            return result
        except CustomException as e:
            raise e

    async def update_user(self, user: User) -> User:
        """
        Actualiza un usuario existente.

        Args:
            user: Usuario a actualizar

        Returns:
            User: Usuario actualizado
        """
        await self.__ensure_session_health()
        logger.info(f"Inicia el flujo de actualización de usuario en base de datos. ID: {user.id}")
        try:
            result = await self.user_repository.update_user(user)
            return result
        except CustomException as e:
            raise e

    async def create_image_metadata(self, image_metadata: ImageMetadata) -> ImageMetadata:
        """
        Crea un nuevo metadato de imagen en la base de datos.

        Args:
            image_metadata (ImageMetadata): Objeto de dominio ImageMetadata a crear.
            
        Returns:
            ImageMetadata: Metadato de imagen creado.
        """
        await self.__ensure_session_health()
        logger.info("Inicia el flujo de creación de metadatos de imagen en base de datos")
        logger.info(f"Metadatos de imagen a crear: {image_metadata}")
        
        # Asegurarse de que todos los campos de texto tengan valores adecuados
        if image_metadata.weather_tags is None:
            image_metadata.weather_tags = ""
        
        if image_metadata.uploader_username is None:
            image_metadata.uploader_username = ""
            
        if image_metadata.image_url is None:
            image_metadata.image_url = ""
            
        if image_metadata.image_name is None:
            image_metadata.image_name = ""
        
        try:
            image_metadata_entity = ImageMapper.map_image_metadata_to_entity(image_metadata)
            result = await self.image_repository.create_image_metadata(image_metadata_entity)
            return result
        except CustomException as e:
            raise e
        except SQLAlchemyError as e:
            logger.error(f"Error al crear metadatos de imagen: {e}")
            self.session.rollback()
            await asyncio.sleep(0.5)  # Esperar antes de lanzar la excepción
            raise CustomException(ResponseCodeEnum.KOG02)

    async def get_total_images_uploaded(self) -> int:
        """
        Obtiene el total de imágenes subidas por los usuarios.

        Returns:
            int: Total de imágenes subidas
        """
        await self.__ensure_session_health()
        try:
            return await self.masterdata_repository.get_total_images_uploaded()
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener el total de imágenes subidas: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)
        
    async def get_top_users_by_images_uploaded_with_count(self) -> List[Tuple[User, int]]:
        """
        Obtiene los usuarios con más imágenes subidas junto con el conteo de imágenes.

        Returns:
            list[tuple[User, int]]: Lista de tuplas con usuario y cantidad de imágenes
        """
        await self.__ensure_session_health()
        try:
            result = await self.masterdata_repository.get_top_users_by_images_uploaded()
            users_with_count = []
            for user_entity, total_images in result:
                user = mapper.map_entity_to_user(user_entity)
                users_with_count.append((user, total_images))
            return users_with_count
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener los usuarios con más imágenes subidas y su conteo: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)
    
    async def get_total_images_uploaded_by_user(self, username: str) -> int:
        """
        Obtiene el total de imágenes subidas por un usuario.

        Args:
            username (str): Nombre de usuario del usuario a buscar.

        Returns:
            int: Total de imágenes subidas
        """
        await self.__ensure_session_health()
        try:
            return await self.masterdata_repository.get_total_images_uploaded_by_user(username)
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener el total de imágenes subidas por usuario: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)   