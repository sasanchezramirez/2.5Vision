import logging
from typing import Final

from app.domain.gateway.persistence_gateway import PersistenceGateway
from app.domain.model.user import UserContribution

logger: Final[logging.Logger] = logging.getLogger("MasterData UseCase")

class MasterdataUseCase:
    def __init__(self, persistence_gateway: PersistenceGateway):
        self.persistence_gateway = persistence_gateway

    def get_total_images_uploaded(self) -> int:
        """
        Obtiene el total de imágenes subidas al sistema por todos los usuarios.
        """
        return self.persistence_gateway.get_total_images_uploaded()
    
    def get_top_users_by_images_uploaded(self) -> list[UserContribution]:
        """
        Obtiene el top 3 de usuarios con más imágenes subidas y su contribución total.
        """
        result = self.persistence_gateway.get_top_users_by_images_uploaded_with_count()
        top_users_contribution = []
        
        for user, total_images in result:
            top_users_contribution.append(
                UserContribution(
                    id=user.id, 
                    username=user.username, 
                    total_images_uploaded=total_images
                )
            )
        return top_users_contribution
    
    def get_total_images_uploaded_by_user(self, username: str) -> int:
        """
        Obtiene el total de imágenes subidas por un usuario.
        
        Args:
            username: Nombre de usuario del usuario
        """
        return self.persistence_gateway.get_total_images_uploaded_by_user(username)
