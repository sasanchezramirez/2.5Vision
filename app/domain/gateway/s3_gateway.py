from abc import ABC, abstractmethod
from fastapi import UploadFile
from typing import Dict, Any

class S3Gateway(ABC):
    @abstractmethod
    def upload_image(self, file: UploadFile) -> Dict[str, Any]:
        """
        Sube una imagen a un bucket de AWS S3.
        Args:
            file: Archivo de imagen a subir

        Returns:
            Dict[str, Any]: Información sobre la imagen subida (URL, nombre, tipo, tamaño)
        """
        pass
