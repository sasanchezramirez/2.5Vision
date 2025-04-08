from abc import ABC, abstractmethod
from typing import Optional

from app.domain.model.user import User
from app.domain.model.image_metadata import ImageMetadata

class PersistenceGateway(ABC):
    """
    Interfaz abstracta para el gateway de persistencia.
    
    Esta interfaz define los métodos que deben implementar las clases
    concretas que manejen la persistencia de datos, siguiendo el principio
    de inversión de dependencias de la arquitectura hexagonal.
    """

    @abstractmethod
    def create_user(self, user: User) -> User:
        """
        Crea un nuevo usuario en la base de datos.

        Args:
            user: Usuario a crear

        Returns:
            User: Usuario creado con ID asignado

        Raises:
            CustomException: Si hay un error al crear el usuario
        """
        pass

    @abstractmethod
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Obtiene un usuario por su ID.

        Args:
            user_id: ID del usuario a buscar

        Returns:
            Optional[User]: Usuario encontrado o None si no existe

        Raises:
            CustomException: Si hay un error al obtener el usuario
        """
        pass

    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Obtiene un usuario por su correo electrónico.

        Args:
            email: Correo electrónico del usuario a buscar

        Returns:
            Optional[User]: Usuario encontrado o None si no existe

        Raises:
            CustomException: Si hay un error al obtener el usuario
        """
        pass

    @abstractmethod
    def update_user(self, user: User) -> User:
        """
        Actualiza los datos de un usuario existente.

        Args:
            user: Usuario con los datos a actualizar

        Returns:
            User: Usuario actualizado

        Raises:
            CustomException: Si hay un error al actualizar el usuario
        """
        pass

    @abstractmethod
    def create_image_metadata(self, image_metadata: ImageMetadata) -> ImageMetadata:
        """
        Crea un nuevo metadato de imagen en la base de datos.

        Args:
            image_metadata: Metadato de imagen a crear

        Returns:
            ImageMetadata: Metadato de imagen creado
        """
        pass    
    
