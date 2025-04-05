import logging
import boto3
from app.domain.gateway.s3_gateway import S3Gateway
from app.application.settings import settings
from typing import Final, Dict, Any
from fastapi import UploadFile
import os

logger: Final[logging.Logger] = logging.getLogger("S3 Service")

class S3Service(S3Gateway):
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION)   
    
    def upload_image(self, file: UploadFile) -> Dict[str, Any]:
        """
        Sube una imagen a un bucket de AWS S3.
        Args:
            file: Archivo de imagen a subir

        Returns:
            Dict[str, Any]: Información sobre la imagen subida (URL, nombre, tipo, tamaño)    
        """
        logger.info("Inicia conexión con S3")
        path = f"images/{file.filename}"
        
        try:
            # Guardar temporalmente el contenido del archivo para obtener su tamaño
            file_content = file.file.read()
            file_size = len(file_content)
            
            # Regresamos al inicio del archivo para poder subirlo
            file.file.seek(0)
            
            # Subir archivo a S3
            self.s3.upload_fileobj(
                file.file,
                settings.AWS_S3_BUCKET,
                path
            )
            
            # Construir la URL de la imagen
            if settings.ENV == "local" or settings.ENV == "development":
                base_url = f"https://{settings.AWS_S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com"
            else:
                # URL personalizada para producción si la tienes
                base_url = f"https://{settings.AWS_S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com"
                
            image_url = f"{base_url}/{path}"
            
            return {
                "image_url": image_url,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file_size
            }
        except Exception as e:
            logger.error(f"Error al subir la imagen a S3: {e}")
            raise e