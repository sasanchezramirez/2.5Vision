import logging
from typing import Optional, Final

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

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
    Implementación del gateway de persistencia que maneja las operaciones de base de datos.
    
    Esta clase implementa la interfaz PersistenceGateway y proporciona métodos para
    realizar operaciones CRUD en la base de datos utilizando SQLAlchemy.
    
    Attributes:
        session (Session): Sesión de SQLAlchemy para operaciones de base de datos.
        user_repository (UserRepository): Repositorio para operaciones específicas de usuarios.
    """
    
    def __init__(self, session: Session) -> None:
        """
        Inicializa el servicio de persistencia.
        
        Args:
            session (Session): Sesión de SQLAlchemy para operaciones de base de datos.
        """
        logger.info("Inicializando servicio de persistencia")
        self.session: Final[Session] = session
        self.user_repository: Final[UserRepository] = UserRepository(session)
        self.image_repository: Final[ImageRepository] = ImageRepository(session)
        self.masterdata_repository: Final[MasterdataRepository] = MasterdataRepository(session)

    def create_user(self, user: User) -> User:
        """
        Crea un nuevo usuario en la base de datos.
        
        Args:
            user (User): Objeto de dominio User a crear.
            
        Returns:
            User: Usuario creado con su ID asignado.
            
        Raises:
            CustomException: Si hay un error en la validación o en la operación de base de datos.
        """
        try:
            # Verificar si el nombre de usuario ya existe
            existing_user = self.user_repository.get_user_by_username(user.username)
            if existing_user:
                raise CustomException(ResponseCodeEnum.KOU01)
                
            user_entity = UserEntity.from_user(user)
            created_user_entity = self.user_repository.create_user(user_entity)
            self.session.commit()
            return mapper.map_entity_to_user(created_user_entity)
        except CustomException as e:
            self.session.rollback()
            raise e
        except IntegrityError as e:
            logger.error(f"Error de integridad al crear usuario: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOU01)
        except SQLAlchemyError as e:
            logger.error(f"Error al crear usuario: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)
        
    def get_user_by_id(self, id: int) -> Optional[User]:
        """
        Obtiene un usuario por su ID.
        
        Args:
            id (int): ID del usuario a buscar.
            
        Returns:
            Optional[User]: Usuario encontrado o None si no existe.
            
        Raises:
            CustomException: Si hay un error en la operación de base de datos.
        """
        try:
            user_entity = self.user_repository.get_user_by_id(id)
            return mapper.map_entity_to_user(user_entity)
        except CustomException as e:
            raise e
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener usuario: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)
        
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Obtiene un usuario por su nombre de usuario.
        
        Args:
            username (str): Nombre de usuario del usuario a buscar.
            
        Returns:
            Optional[User]: Usuario encontrado o None si no existe.
            
        Raises:
            CustomException: Si hay un error en la operación de base de datos.
        """
        try:
            user_entity = self.user_repository.get_user_by_username(username)
            if not user_entity:
                raise CustomException(ResponseCodeEnum.KOU02)
            return mapper.map_entity_to_user(user_entity)
        except CustomException as e:
            raise e
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener usuario: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)
    
    def update_user(self, user: User) -> User:
        """
        Actualiza un usuario existente en la base de datos.
        
        Args:
            user (User): Objeto de dominio User con los datos actualizados.
            
        Returns:
            User: Usuario actualizado.
            
        Raises:
            CustomException: Si el usuario no existe o hay un error en la operación.
        """
        try:
            existing_user = self.user_repository.get_user_by_id(user.id)
            if not existing_user:
                raise CustomException(ResponseCodeEnum.KOU02)
            user_entity = mapper.map_update_to_entity(user, existing_user)
            updated_user_entity = self.user_repository.update_user(user_entity)
            self.session.commit()
            return mapper.map_entity_to_user(updated_user_entity)
        except CustomException as e:
            self.session.rollback()
            raise e
        except SQLAlchemyError as e:
            logger.error(f"Error al actualizar usuario: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)
    
    def create_image_metadata(self, image_metadata: ImageMetadata) -> ImageMetadata:
        """
        Crea un nuevo metadato de imagen en la base de datos.

        Args:
            image_metadata (ImageMetadata): Objeto de dominio ImageMetadata a crear.
            
        Returns:
            ImageMetadata: Metadato de imagen creado.
        """
        logger.info("Inicia el flujo de creación de metadatos de imagen en base de datos")
        logger.info(f"Metadatos de imagen a crear: {image_metadata}")
        try:
            image_metadata_entity = ImageMapper.map_image_metadata_to_entity(image_metadata)
            self.image_repository.create_image_metadata(image_metadata_entity)
            self.session.commit()
            return image_metadata
        except CustomException as e:
            self.session.rollback()
            raise e
        except SQLAlchemyError as e:
            logger.error(f"Error al crear metadatos de imagen: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)

    def get_total_images_uploaded(self) -> int:
        """
        Obtiene el total de imágenes subidas por los usuarios.

        Returns:
            int: Total de imágenes subidas
        """
        try:
            return self.masterdata_repository.get_total_images_uploaded()
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener el total de imágenes subidas: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)
        
 
    
    def get_top_users_by_images_uploaded_with_count(self) -> list[tuple[User, int]]:
        """
        Obtiene los usuarios con más imágenes subidas junto con el conteo de imágenes.

        Returns:
            list[tuple[User, int]]: Lista de tuplas con usuario y cantidad de imágenes
        """
        try:
            result = self.masterdata_repository.get_top_users_by_images_uploaded()
            users_with_count = []
            for user_entity, total_images in result:
                user = mapper.map_entity_to_user(user_entity)
                users_with_count.append((user, total_images))
            return users_with_count
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener los usuarios con más imágenes subidas y su conteo: {e}")
            self.session.rollback()
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
            return self.masterdata_repository.get_total_images_uploaded_by_user(username)
        except SQLAlchemyError as e:
            logger.error(f"Error al obtener el total de imágenes subidas por usuario: {e}")
            self.session.rollback()
            raise CustomException(ResponseCodeEnum.KOG02)   