import logging
from typing import Final
from PIL import Image
logger: Final[logging.Logger] = logging.getLogger("Geolocalization Utils")

class GeolocalizationUtils:
    @staticmethod
    def get_zone(latitude: float, longitude: float) -> int:
        """
        Obtiene la zona geográfica de la imagen.

        Args:
            latitude: Latitud de la imagen
            longitude: Longitud de la imagen

        Returns:
            int: Zona de procesamiento
        """
        pass

    @staticmethod
    def dms_to_decimal(dms: tuple[int, int, float], ref: str) -> float:
        """
        Convierte coordenadas DMS a decimales.

        Args:
            dms: Coordenadas DMS
            ref: Referencia de la coordenada

        Returns:
            float: Coordenadas decimales
        """
        degrees, minutes, seconds = dms
        decimal = degrees[0] / degrees[1] + \
                minutes[0] / minutes[1] / 60 + \
                seconds[0] / seconds[1] / 3600
        if ref in ['S', 'W']:
            decimal *= -1
        return decimal