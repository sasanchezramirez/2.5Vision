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
    def dms_to_decimal(dms: tuple, ref: str) -> float:
        """
        Convierte coordenadas DMS a decimales.

        Args:
            dms: Coordenadas DMS (grados, minutos, segundos)
            ref: Referencia de la coordenada (N, S, E, W)

        Returns:
            float: Coordenadas decimales
        """
        degrees, minutes, seconds = dms
        
        # Comprobar si son objetos IFDRational (tienen atributos numerator y denominator)
        if hasattr(degrees, 'numerator') and hasattr(degrees, 'denominator'):
            degrees_value = degrees.numerator / degrees.denominator
            minutes_value = minutes.numerator / minutes.denominator
            seconds_value = seconds.numerator / seconds.denominator
        else:
            # Si son tuplas como (numerador, denominador)
            try:
                degrees_value = degrees[0] / degrees[1]
                minutes_value = minutes[0] / minutes[1]
                seconds_value = seconds[0] / seconds[1]
            except (TypeError, IndexError):
                # Si son valores directos (float o int)
                degrees_value = float(degrees)
                minutes_value = float(minutes)
                seconds_value = float(seconds)
        
        decimal = degrees_value + minutes_value / 60 + seconds_value / 3600
        if ref in ['S', 'W']:
            decimal *= -1
        return decimal