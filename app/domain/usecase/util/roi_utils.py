import logging
from typing import Final
from PIL import Image
logger: Final[logging.Logger] = logging.getLogger("ROI Utils")

class ROIUtils:
    @staticmethod
    def get_roi(image: Image) -> Image:
        """
        Obtiene la región de interés de la imagen.

        Args:
            image: Imagen a obtener la región de interés

        Returns:
            Image: Región de interés de la imagen
        """
        pass

    @staticmethod
    def get_pixels(image: Image) -> list[int]:
        """
        Obtiene los valores de los píxeles de la imagen.

        Args:
            image: Imagen a obtener los valores de los píxeles

        Returns:
            list[int]: Valores de los píxeles de la imagen
        """
        pass
