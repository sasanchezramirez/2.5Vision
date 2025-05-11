import logging
import boto3
import aioboto3
import asyncio
from app.domain.gateway.s3_gateway import S3Gateway
from app.application.settings import settings
from typing import Final
from app.infrastructure.driven_adapter.s3.dto.upload_image_response import UploadImageResponse
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
from PIL import Image
from datetime import datetime
import io 
logger: Final[logging.Logger] = logging.getLogger("S3 Service")

class S3Service(S3Gateway):
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION)   
    
    def upload_image(self, image: Image, has_metadata: bool) -> UploadImageResponse:
        """
        Sube una imagen a un bucket de AWS S3.
        Args:
            file: Archivo de imagen a subir
            has_metadata: Indica si la imagen tiene metadatos
        Returns:
            UploadImageResponse: Respuesta con la URL de la imagen subida y algunos datos de la imagen
        """
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
        logger.info("Inicia conexión con S3")
        if has_metadata:
            path = f"images/with_metadata/{filename}"
        else:
            path = f"images/no_metadata/{filename}"
        
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)  

            self.s3.upload_fileobj(
                img_byte_arr,
                settings.AWS_S3_BUCKET,
                path
            )
            
            base_url = f"https://{settings.AWS_S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com"
                
            image_url = f"{base_url}/{path}"

            logger.info(f"Imagen subida a S3: {image_url}")
            
            return UploadImageResponse(
                image_url=image_url,
                image_name=filename
      
            )
        except Exception as e:
            logger.error(f"Error al subir la imagen a S3: {e}")
            raise CustomException(ResponseCodeEnum.VIM02)
    
    async def upload_image_async(self, image: Image, has_metadata: bool) -> UploadImageResponse:
        """
        Sube una imagen a un bucket de AWS S3 de forma asíncrona.
        Args:
            image: Imagen a subir (objeto PIL.Image)
            has_metadata: Indica si la imagen tiene metadatos
        Returns:
            UploadImageResponse: Respuesta con la URL de la imagen subida y algunos datos de la imagen
        """
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
        logger.info("Inicia conexión asíncrona con S3")
        if has_metadata:
            path = f"images/with_metadata/{filename}"
        else:
            path = f"images/no_metadata/{filename}"
        
        try:
            # Preparar la imagen
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Crear sesión asíncrona de boto3
            session = aioboto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            
            # Subir la imagen de forma asíncrona
            async with session.client('s3') as s3_client:
                await s3_client.upload_fileobj(
                    img_byte_arr,
                    settings.AWS_S3_BUCKET,
                    path
                )
            
            base_url = f"https://{settings.AWS_S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com"
            image_url = f"{base_url}/{path}"
            
            logger.info(f"Imagen subida de forma asíncrona a S3: {image_url}")
            
            return UploadImageResponse(
                image_url=image_url,
                image_name=filename
            )
            
        except Exception as e:
            logger.error(f"Error al subir la imagen a S3 de forma asíncrona: {e}")
            raise CustomException(ResponseCodeEnum.VIM02)