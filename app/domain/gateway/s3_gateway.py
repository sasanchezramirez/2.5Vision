from abc import ABC, abstractmethod
from fastapi import UploadFile
class S3Gateway(ABC):
    @abstractmethod
    def upload_image(self, file: UploadFile) -> bool:
        """
        Sube una imagen a un bucket de AWS S3.
        Args:
            file: Archivo de imagen a subir

        Returns:
            bool: Respuesta con el resultado de la operación
        """
        pass
