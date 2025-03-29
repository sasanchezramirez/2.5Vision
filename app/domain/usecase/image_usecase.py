import logging
from typing import Final
from fastapi import UploadFile
from PIL import Image

from app.domain.usecase.util.image_processing_utils import ImageProcessingUtils

logger: Final[logging.Logger] = logging.getLogger("Image UseCase")

class ImageUseCase:
    def __init__(self):
        pass

    async def execute(self, file: UploadFile) -> dict:
        """
        Pipelin de procesamiento de imagen para detectar la cantidad de material particulado presente.
        
        Args:
            file: Archivo de imagen a subir
            
        Returns:
            dict: Estado del proceso
        """
        normalized_image = self._image_normalization(file)

    def _image_normalization(self, image: Image) -> Image:
        """
        Normaliza la imagen como preprocesamiento para el modelo de detección de material particulado.
        - Ajusta el contraste
        - Ajusta el balance de blancos
        - Reduce el ruido
    
        Args:
            image: Imagen a normalizar
            
        Returns:
            Image: Imagen normalizada
        """
        exposure_corrected_image = ImageProcessingUtils.exposure_correction(image)
        white_balanced_image = ImageProcessingUtils.white_balance_correction(exposure_corrected_image)
        noise_reduced_image = ImageProcessingUtils.noise_reduction(white_balanced_image)
        return noise_reduced_image
    