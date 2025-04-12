import logging
from typing import Final, Tuple
import io
from PIL import Image, ImageEnhance, ImageStat
import numpy as np
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
from fastapi import UploadFile
from PIL.ExifTags import TAGS, GPSTAGS
from app.domain.model.image_config_metadata import ImageConfigMetadata
from math import log2

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
    def exposure_correction(image: Image, image_config_metadata: ImageConfigMetadata) -> Image:
        """
        Ajusta el contraste de la imagen para que los detalles sean más visibles. Esto se hace usando los metadatos de la imagen
        y estableciendo un valor objetivo de estandarizacion para la exposición de cada una de las imágenes
        
        Args:
            image: Imagen a ajustar
            image_config_metadata: Metadatos de configuración de la imagen
            
        Returns:
            Image: Imagen con contraste ajustado
        """
        logger.info("Inicia el flujo de corrección de exposición a la imagen")
        iso = image_config_metadata.iso
        shutter_speed = image_config_metadata.shutter_speed
        aperture = image_config_metadata.aperture
        exposure_compensation = image_config_metadata.exposure_compensation

        # Usar 0.0 como valor por defecto para exposure_compensation si es None
        if exposure_compensation is None:
            exposure_compensation = 0.0

        if iso and shutter_speed and aperture:
            current_ev = (log2(aperture**2 / shutter_speed)
                        - log2(iso / 100)
                        - exposure_compensation)
            target_ev = 13.0
            delta_ev = target_ev - current_ev

            #Aquí se aplica el factor de escala
            logger.info(f"Aplica el factor de escala a la imagen: {delta_ev}")
            scale = 2**delta_ev
            enhancer = ImageEnhance.Brightness(image)
            corrected_image = enhancer.enhance(scale)
            return corrected_image
        else:
            logger.info("No se pudo aplicar la corrección de exposición a la imagen por ausencia de metadatos")
            stat = ImageStat.Stat(image)
            mean_lum = sum(stat.mean) / len(stat.mean)  

            desired_mean = 128 
            if mean_lum < 1: mean_lum = 1
            scale = desired_mean / mean_lum

            enhancer = ImageEnhance.Brightness(image)
            corrected_image = enhancer.enhance(scale)
            return corrected_image


    @staticmethod
    def white_balance_correction(image: Image) -> Image:
        """
        Ajusta el balance de blancos de la imagen para que los colores sean más naturales.
        Para esto primero se calcula el promedio de cada canal de la imagen,
        luego se asume un valor de gris neutro para cada canal y se calcula el factor de corrección
        para cada canal.
        Args:
            image: Imagen a ajustar
            
        Returns:
            Image: Imagen con balance de blancos ajustado
        """

        logger.info("Inicia el flujo de corrección de balance de blancos a la imagen")
        img_array = np.array(image).astype(float)


        mean_r = np.mean(img_array[:, :, 0])
        mean_g = np.mean(img_array[:, :, 1])
        mean_b = np.mean(img_array[:, :, 2])

        mean_gray = (mean_r + mean_g + mean_b) / 3


        factor_r = mean_gray / mean_r if mean_r != 0 else 1
        factor_g = mean_gray / mean_g if mean_g != 0 else 1
        factor_b = mean_gray / mean_b if mean_b != 0 else 1

        # Aquí se corrección multiplicativa
        img_array[:, :, 0] *= factor_r
        img_array[:, :, 1] *= factor_g
        img_array[:, :, 2] *= factor_b

        # Clip a [0, 255] y convertir a uint8. 
        # Esto es para que los valores no se salgan de los límites de 0 a 255
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        logger.info("Finaliza el flujo de corrección de balance de blancos a la imagen")
        return Image.fromarray(img_array)
    
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
    

