import logging
from typing import Final, List, Optional
import time

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import asyncio
import psycopg2
import psycopg2.extras
from datetime import datetime
from app.application.settings import settings

from app.domain.model.image_metadata import ImageMetadata
from app.infrastructure.driven_adapter.persistence.mapper.image_mapper import ImageMapper
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
from app.infrastructure.driven_adapter.persistence.entity.image_metadata_entity import ImageMetadataEntity
from app.infrastructure.driven_adapter.persistence.config.database import retry_on_db_error

logger: Final[logging.Logger] = logging.getLogger("Image Repository")

class ImageRepository:
    """
    Implementación del repositorio de imágenes.
    
    Esta clase implementa las operaciones de persistencia para imágenes
    utilizando SQLAlchemy como ORM.
    """
    def __init__(self, session: Session):
        self.session: Final[Session] = session
        
    async def create_image_metadata(self, image_metadata_entity: ImageMetadataEntity) -> ImageMetadata:
        """
        Crea un nuevo metadato de imagen en la base de datos. 
        Aquí decidí usar una conexión directa a PostgreSQL para evitar problemas de sincronización
        con la sesión de SQLAlchemy y para mejorar el rendimiento.

        Args:
            image_metadata: Metadato de imagen a crear

        Returns:
            ImageMetadata: Metadato de imagen creado
        """
        logger.info(f"Creando metadato de imagen: {image_metadata_entity}")
        
        connection = None
        
        weather_tags = image_metadata_entity.weather_tags if image_metadata_entity.weather_tags is not None else ""
        uploader_username = image_metadata_entity.uploader_username if image_metadata_entity.uploader_username is not None else ""
        image_url = image_metadata_entity.image_url if image_metadata_entity.image_url is not None else ""
        image_name = image_metadata_entity.image_name if image_metadata_entity.image_name is not None else ""
        
        latitude = image_metadata_entity.latitude
        if latitude is not None and not isinstance(latitude, float):
            latitude = float(latitude)
        
        longitude = image_metadata_entity.longitude
        if longitude is not None and not isinstance(longitude, float):
            longitude = float(longitude)
        
        visibility_score = image_metadata_entity.visibility_score
        if not isinstance(visibility_score, int):
            visibility_score = int(visibility_score)
            
        try:
            logger.info("Usando conexión directa a PostgreSQL...")
            connection = psycopg2.connect(settings.DATABASE_URL)
            cursor = connection.cursor()
            
            query = """
            INSERT INTO vision_2_5.image_metadata 
            (latitude, longitude, datetime_taken, visibility_score, weather_tags, uploader_username, image_url, image_name) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """
            
            def execute_query():
                try:
                    cursor.execute(
                        query,
                        (
                            latitude,
                            longitude,
                            image_metadata_entity.datetime_taken,
                            visibility_score,
                            weather_tags,
                            uploader_username,
                            image_url,
                            image_name
                        )
                    )
                    result = cursor.fetchone()
                    connection.commit()
                    return result[0] if result else None
                except Exception as e:
                    logger.error(f"Error en ejecución directa: {e}")
                    connection.rollback()
                    raise e
            
            loop = asyncio.get_event_loop()
            image_id = await loop.run_in_executor(None, execute_query)
            
            if image_id:
                logger.info(f"Imagen creada con éxito, ID: {image_id}")
                image_metadata_entity.id = image_id
                
                return ImageMetadata(
                    latitude=latitude,
                    longitude=longitude,
                    datetime_taken=image_metadata_entity.datetime_taken,
                    visibility_score=visibility_score,
                    weather_tags=weather_tags,
                    uploader_username=uploader_username,
                    image_url=image_url,
                    image_name=image_name
                )
            else:
                logger.error("No se pudo obtener el ID de la imagen creada")
                raise CustomException(ResponseCodeEnum.KOG02)
                
        except Exception as e:
            logger.error(f"Error en la creación directa de metadatos: {e}")
            if connection:
                try:
                    connection.rollback()
                except:
                    pass
            raise CustomException(ResponseCodeEnum.KOG02)
        finally:
            if connection:
                try:
                    connection.close()
                except:
                    pass
        
        
        