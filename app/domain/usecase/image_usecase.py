import logging
from typing import Final
from fastapi import UploadFile

logger: Final[logging.Logger] = logging.getLogger("Image UseCase")

class ImageUseCase:
    async def upload_image(self, file: UploadFile) -> dict:
        """
        Procesa y sube una imagen al servidor.
        
        Args:
            file: Archivo de imagen a subir
            
        Returns:
            dict: Información de la imagen subida
        """
        pass 