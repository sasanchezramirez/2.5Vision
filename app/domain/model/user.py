from pydantic import BaseModel, Field
from typing import Optional


class User(BaseModel):
    """
    Modelo de dominio para representar un usuario en el sistema.
    
    Este modelo define la estructura de datos y las validaciones
    necesarias para un usuario en el sistema.
    """
    
    id: Optional[int] = Field(
        default=None,
        description="Identificador único del usuario"
    )
    
    username: str = Field(
        description="Nombre de usuario del usuario",
        example="usuario"
    )
    
    password: Optional[str] = Field(
        default=None,
        description="Contraseña hasheada del usuario",
        min_length=8
    )
    
    creation_date: Optional[str] = Field(
        default=None,
        description="Fecha de creación del usuario en formato ISO"
    )
    
    profile_id: Optional[int] = Field(
        default=None,
        description="Identificador del perfil del usuario",
        gt=0
    )
    
    status_id: Optional[int] = Field(
        default=None,
        description="Identificador del estado del usuario",
        gt=0
    )
   
    contact_info: Optional[str] = Field(
        default=None,
        description="Información de contacto del usuario",
        example="usuario@example.com"
    )

    class Config:
        """
        Configuración del modelo Pydantic.
        """
        json_schema_extra = {
            "example": {
                "id": 1,
                "username": "usuario",
                "creation_date": "2024-03-27T12:00:00",
                "profile_id": 1,
                "status_id": 1
            }
        }

class UserContribution(BaseModel):
    """
    Modelo de dominio para representar la contribución de un usuario en el sistema.
    """
    id: int = Field(
        description="Identificador único del usuario"
    )
    username: str = Field(
        description="Nombre de usuario del usuario"
    )
    total_images_uploaded: int = Field(
        description="Total de imágenes subidas por el usuario"
    )

    def model_dump(self) -> dict:
        """
        Convierte el modelo a un diccionario serializable.
        
        Returns:
            dict: Diccionario con los datos del modelo
        """
        return {
            "id": self.id,
            "username": self.username,
            "total_images_uploaded": self.total_images_uploaded
        }

    # Para compatibilidad con versiones anteriores
    def dict(self) -> dict:
        """
        Alias para model_dump para compatibilidad con Pydantic v1.
        """
        return self.model_dump()

    class Config:
        """
        Configuración del modelo Pydantic.
        """
        json_schema_extra = {
            "example": {
                "id": 1,
                "username": "usuario",
                "total_images_uploaded": 10
            }
        }
        