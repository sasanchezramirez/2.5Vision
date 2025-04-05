import logging
import boto3
from app.domain.gateway.s3_gateway import S3Gateway
from app.application.settings import settings
from typing import Final
from fastapi import UploadFile

logger: Final[logging.Logger] = logging.getLogger("S3 Service")

class S3Service(S3Gateway):
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            bucket_name=settings.AWS_S3_BUCKET)   
    
    def upload_image(self, file: UploadFile) -> bool:
        """
        Sube una imagen a un bucket de AWS S3.
        Args:
            file: Archivo de imagen a subir

        Returns:
            bool: Respuesta con el resultado de la operación    
        """
        logger.info("Inicia conexión con S3")
        path = f"images/{file.filename}"
        try:
            self.s3.upload_fileobj(
                file.file,
                settings.AWS_S3_BUCKET,
                path
            )
            return True
        except Exception as e:
            logger.error(f"Error al subir la imagen a S3: {e}")
            return False