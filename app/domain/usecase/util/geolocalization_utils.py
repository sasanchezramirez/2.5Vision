import logging
from typing import Final
from PIL import Image
import math
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
        
        # Verificar valores NaN
        if (isinstance(degrees, float) and math.isnan(degrees)) or \
           (isinstance(minutes, float) and math.isnan(minutes)) or \
           (isinstance(seconds, float) and math.isnan(seconds)):
            logger.warning("Valores NaN detectados en las coordenadas DMS")
            return None
        
        # Comprobar si son objetos IFDRational (tienen atributos numerator y denominator)
        if hasattr(degrees, 'numerator') and hasattr(degrees, 'denominator'):
            if degrees.denominator == 0 or minutes.denominator == 0 or seconds.denominator == 0:
                logger.warning("División por cero detectada en coordenadas DMS")
                return None
            degrees_value = degrees.numerator / degrees.denominator
            minutes_value = minutes.numerator / minutes.denominator
            seconds_value = seconds.numerator / seconds.denominator
        else:
            # Si son tuplas como (numerador, denominador)
            try:
                if degrees[1] == 0 or minutes[1] == 0 or seconds[1] == 0:
                    logger.warning("División por cero detectada en coordenadas DMS")
                    return None
                degrees_value = degrees[0] / degrees[1]
                minutes_value = minutes[0] / minutes[1]
                seconds_value = seconds[0] / seconds[1]
            except (TypeError, IndexError, ZeroDivisionError):
                # Si son valores directos (float o int)
                try:
                    degrees_value = float(degrees)
                    minutes_value = float(minutes)
                    seconds_value = float(seconds)
                except (TypeError, ValueError):
                    logger.warning("No se pudieron convertir valores DMS a float")
                    return None
        
        decimal = degrees_value + minutes_value / 60 + seconds_value / 3600
        if ref in ['S', 'W']:
            decimal *= -1
        return decimal