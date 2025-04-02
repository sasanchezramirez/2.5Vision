from pydantic import BaseModel, Field
from typing import Optional

class GPS(BaseModel):
    """
    Modelo de dominio para representar las coordenadas GPS de una imagen.
    """
    latitude: Optional[float] = Field(
        default=None,
        description="Latitud de la imagen"
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Longitud de la imagen"
    )
    zone: int = Field(
        default=0,
        description="Zona geográfica preestablecida en el área metropolitana de Medellín"
    )
