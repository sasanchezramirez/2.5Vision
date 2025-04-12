from pydantic import BaseModel, Field
from typing import Optional, Union
from datetime import datetime

class ImageConfigMetadata(BaseModel):
    camera_make: Optional[str] = Field(
        None, 
        description="Marca del dispositivo o cámara (ej. Canon, Nikon, iPhone...)."
    )
    camera_model: Optional[str] = Field(
        None, 
        description="Modelo del dispositivo o cámara (ej. EOS 5D Mark IV, Pixel 6...)."
    )
    iso: Optional[int] = Field(
        None,
        description="Valor de sensibilidad ISO."
    )
    shutter_speed: Optional[float] = Field(
        None,
        description="Tiempo de exposición (shutter speed) en segundos."
    )
    aperture: Optional[float] = Field(
        None,
        description="Apertura del diafragma (f-number)."
    )
    exposure_compensation: Optional[float] = Field(
        None,
        description="Compensación de exposición en pasos EV."
    )
    focal_length: Optional[float] = Field(
        None,
        description="Longitud focal en milímetros (mm)."
    )
    datetime_original: Optional[Union[datetime, str]] = Field(
        None,
        description="Fecha y hora originales en que se tomó la foto. Puede ser un objeto datetime o una cadena en formato 'YYYY-MM-DD HH:MM:SS'."
    )
