import logging
from typing import Final

logger: Final[logging.Logger] = logging.getLogger("Statistical Utils")

class StatisticalUtils:
    @staticmethod
    def get_median(pixel_values: list[int]) -> float:
        """
        Obtiene el valor mediano de los píxeles de la imagen.

        Args:
            pixel_values: Valores de los píxeles de la imagen
            
        Returns:
            int: Valor mediano de los píxeles de la imagen
        """
        pass
    
    def get_mean(pixel_values: list[int]) -> float:
        """
        Obtiene el valor promedio de los píxeles de la imagen.

        Args:
            pixel_values: Valores de los píxeles de la imagen
            
        Returns:
            float: Valor promedio de los píxeles de la imagen
        """
        pass
    
    