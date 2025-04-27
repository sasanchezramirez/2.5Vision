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
