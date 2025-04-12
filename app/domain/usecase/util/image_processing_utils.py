import logging
from typing import Final, Tuple
import io
from PIL import Image
import numpy as np
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
from fastapi import UploadFile
from PIL.ExifTags import TAGS, GPSTAGS

logger: Final[logging.Logger] = logging.getLogger("Image Processing Utils")

class ImageProcessingUtils:
    
    @staticmethod
    async def format_upload_file_to_image(file: UploadFile) -> Image:
        """
        Formatea el archivo de imagen subido a un objeto PIL.Image.

        Args:
            file: Archivo de imagen subido
        """
        file_contents = await file.read()
        file_stream = io.BytesIO(file_contents)
        image = Image.open(file_stream)
        return image

    @staticmethod
    def exposure_correction(image: Image) -> Image:
        """
        Ajusta el contraste de la imagen para que los detalles sean más visibles.
        
        Args:
            image: Imagen a ajustar
            
        Returns:
            Image: Imagen con contraste ajustado
        """
        pass

    @staticmethod
    def white_balance_correction(image: Image) -> Image:
        """
        Ajusta el balance de blancos de la imagen para que los colores sean más naturales.
        
        Args:
            image: Imagen a ajustar
            
        Returns:
            Image: Imagen con balance de blancos ajustado
        """
        pass
    
    @staticmethod
    def noise_reduction(image: Image) -> Image:
        """
        Reduce el ruido de la imagen para que los detalles sean más claros.
        
        Args:
            image: Imagen a reducir el ruido    
            
        Returns:
            Image: Imagen con ruido reducido
        """
        pass
    
    @staticmethod
    def color_mapping(image: Image) -> Image:
        """
        Mapea los colores de la imagen para que los detalles sean más claros.
        
        Args:
            image: Imagen a mapear
            
        Returns:
            Image: Imagen con colores mapeados
        """
        pass

    @staticmethod
    def get_image_metadata(image: Image) -> dict:
        """
        Obtiene los metadatos de la imagen. Realiza un proceso de mapeo 
        de los tags numéricos a los nombres legibles.   

        Args:
            image: Imagen a obtener los metadatos

        Returns:
            dict: Metadatos de la imagen    
        """
        logger.info("Inicia el flujo de obtención de metadatos de la imagen")
        info = image._getexif()
        try:
            exif_data = {}
            if info:
                for tag, value in info.items():
                    # Convertir el tag numérico a un nombre legible
                    decoded = TAGS.get(tag, tag)
                    if decoded == "GPSInfo":
                        gps_data = {}
                        for t in value:
                            sub_decoded = GPSTAGS.get(t, t)
                            gps_data[sub_decoded] = value[t]
                        exif_data["GPSInfo"] = gps_data
                    else:
                        exif_data[decoded] = value
            return exif_data
        except Exception as e:
            logger.error(f"Error al obtener los metadatos de la imagen: {e}")
            raise CustomException(ResponseCodeEnum.VIM01)   
    

