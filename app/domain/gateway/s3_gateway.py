from abc import ABC, abstractmethod
from PIL import Image
from app.infrastructure.driven_adapter.s3.dto.upload_image_response import UploadImageResponse

class S3Gateway(ABC):
    @abstractmethod
    def upload_image(self, image: Image, has_metadata: bool) -> UploadImageResponse:
        """
        Sube una imagen a un bucket de AWS S3.
        Args:
            file: Archivo de imagen a subir
            has_metadata: Indica si la imagen tiene metadatos
        Returns:
            UploadImageResponse: Respuesta con la URL de la imagen subida y algunos datos de la imagen
        """
        pass
