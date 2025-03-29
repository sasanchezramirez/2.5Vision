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
    
    

