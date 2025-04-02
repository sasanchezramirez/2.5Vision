import logging
from typing import Final, Tuple
from PIL import Image
import numpy as np

logger: Final[logging.Logger] = logging.getLogger("Image Processing Utils")

class ImageProcessingUtils:

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
        Obtiene los metadatos de la imagen.

        Args:
            image: Imagen a obtener los metadatos

        Returns:
            dict: Metadatos de la imagen    
        """
        pass
    

