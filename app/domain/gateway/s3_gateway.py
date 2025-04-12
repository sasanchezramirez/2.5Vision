from abc import ABC, abstractmethod
from fastapi import UploadFile
from app.infrastructure.driven_adapter.s3.dto.upload_image_response import UploadImageResponse

class S3Gateway(ABC):
    @abstractmethod
    def upload_image(self, file: UploadFile) -> UploadImageResponse:
        """
        Sube una imagen a un bucket de AWS S3.
        Args:
            file: Archivo de imagen a subir

        Returns:
            UploadImageResponse: Respuesta con la URL de la imagen subida y algunos datos de la imagen
        """
        pass
