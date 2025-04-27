from abc import ABC, abstractmethod
from typing import Optional

from app.domain.model.user import User, UserContribution
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
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Obtiene un usuario por su nombre de usuario.

        Args:
            username: Nombre de usuario del usuario a buscar

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

    @abstractmethod
    def get_total_images_uploaded(self) -> int:
        """
        Obtiene el total de imágenes subidas por los usuarios.

        Returns:
            int: Total de imágenes subidas
        """
        pass

    @abstractmethod
    def get_top_users_by_images_uploaded(self) -> list[User]:
        """
        Obtiene los usuarios con más imágenes subidas.

        Returns:
            list[User]: Lista de usuarios con más imágenes subidas  
        """
        pass
    
    @abstractmethod
    def get_top_users_by_images_uploaded_with_count(self) -> list[tuple[User, int]]:
        """
        Obtiene los usuarios con más imágenes subidas junto con el conteo de imágenes.

        Returns:
            list[tuple[User, int]]: Lista de tuplas con usuario y cantidad de imágenes subidas
        """
        pass
    
    @abstractmethod
    def get_total_images_uploaded_by_user(self, username: str) -> int:
        """
        Obtiene el total de imágenes subidas por un usuario.

        Args:
            username: Nombre de usuario a buscar

        Returns:
            int: Total de imágenes subidas
        """
        pass
