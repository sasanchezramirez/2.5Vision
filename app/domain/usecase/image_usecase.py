import logging
from typing import Final
from fastapi import UploadFile
from PIL import Image
import piexif

from app.domain.usecase.util.image_processing_utils import ImageProcessingUtils
from app.domain.usecase.util.roi_utils import ROIUtils
from app.domain.usecase.util.statistical_utils import StatisticalUtils
from app.domain.usecase.util.geolocalization_utils import GeolocalizationUtils
from app.domain.gateway.siata_gateway import SiataGateway
from app.domain.gateway.purpleair_gateway import PurpleAirGateway
from app.domain.gateway.estimation_ml_model_gateway import EstimationMLModelGateway
from app.domain.gateway.s3_gateway import S3Gateway
from app.domain.gateway.persistence_gateway import PersistenceGateway
from app.domain.model.gps import GPS
from app.domain.model.pm_estimation import PMEstimation
from app.domain.model.zones import ZoneDictionary
from app.domain.model.data_sensors import DataSensor
from app.domain.model.image_metadata import ImageMetadata
logger: Final[logging.Logger] = logging.getLogger("Image UseCase")

class ImageUseCase:
    def __init__(self, siata_gateway: SiataGateway, purpleair_gateway: PurpleAirGateway, estimation_ml_model_gateway: EstimationMLModelGateway, s3_gateway: S3Gateway, persistence_gateway: PersistenceGateway):
        self.siata_gateway = siata_gateway
        self.zone_dictionary = ZoneDictionary()
        self.purpleair_gateway = purpleair_gateway
        self.estimation_ml_model_gateway = estimation_ml_model_gateway
        self.s3_gateway = s3_gateway
        self.persistence_gateway = persistence_gateway
    async def data_pipeline(self, file: UploadFile) -> PMEstimation:
        """
        Pipeline de procesamiento de imagen para detectar la cantidad de material particulado presente.
        - Normaliza la imagen
        - Obtiene los datos de la imagen
        - Obtiene la zona geográfica de la imagen~
        - Obtiene los datos del material particulado de la zona
        - Obtiene la estimación de la cantidad de material particulado presente
        - Obtiene la estimación cualitativa de la cantidad de material particulado presente

        Args:
            file: Archivo de imagen a subir 
            
        Returns:
            PMEstimation: Estimación de la cantidad de material particulado presente
        """
        normalized_image = self._image_normalization(file)
        gps_data = self._get_gps_data(file)
        roi_image = ROIUtils.get_roi(normalized_image)
        feature_vector = self._image_processing(roi_image)
        if gps_data.zone and gps_data.zone != 0:
            pm_data = self._get_pm_data(gps_data)
        pm_quantitative_estimation = self._get_pm_quantitative_estimation(pm_data, feature_vector)
        pm_estimation = self._get_pm_qualitative_estimation(pm_quantitative_estimation)
        return pm_estimation
    
    async def upload_image(self, file: UploadFile, image_metadata: ImageMetadata) -> dict:
        """
        Sube una imagen a un bucket de AWS S3.  

        Args:
            file: Archivo de imagen a subir
            image_metadata: Metadatos de la imagen
        Returns:
            dict: Información sobre la imagen subida (URL, nombre, tipo, tamaño)
        """
        logger.info("Subiendo imagen a S3")
        gps_data = self._get_gps_data(file)
        if gps_data.latitude is None or gps_data.longitude is None:
            raise ValueError("No se pudo obtener la latitud y longitud de la imagen")
        image_metadata.latitude = gps_data.latitude
        image_metadata.longitude = gps_data.longitude  
        normalized_image = self._image_normalization(file)
        image_url = self.s3_gateway.upload_image(normalized_image)
        image_metadata.image_url = image_url
        self.persistence_gateway.create_image_metadata(image_metadata)
          
        return image_metadata
    
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
    
    def _image_processing(self, image: Image) -> list[float]:
        """
        Procesa la imagen para detectar la cantidad de material particulado presente.
        - Mapea los colores de la imagen
        - Obtiene los píxeles de la imagen
        - Obtiene la mediana y la media de los píxeles
        - Devuelve un vector de características

        Args:
            image: Imagen a procesar
            
        Returns:
            list[float]: Vector de características
        """
        color_mapping_image = ImageProcessingUtils.color_mapping(image)
        pixel_values = ROIUtils.get_pixels(color_mapping_image)
        median = StatisticalUtils.get_median(pixel_values)
        mean = StatisticalUtils.get_mean(pixel_values)
        feature_vector = [median, mean]
        return feature_vector

    def _get_gps_data(self, image: Image) -> GPS:
        """
        Obtiene los datos de la imagen.
        - Obtiene los datos de la imagen
        - Obtiene la latitud y la longitud de la imagen
        - Obtiene la zona geográfica de la imagen

        Args:
            image: Imagen a verificar

        Returns:
            GPS: Datos de la imagen
        """
        image_metadata = ImageProcessingUtils.get_image_metadata(image)
        gps = image_metadata["GPS"]
        if not gps:
            return GPS(latitude=None, longitude=None, zone=0)
        gps_latitude = gps["latitude"]
        gps_latitude_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef).decode()
        gps_longitude = gps["longitude"]
        gps_longitude_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef).decode()
        latitude = GeolocalizationUtils.dms_to_decimal(gps_latitude, gps_latitude_ref)
        longitude = GeolocalizationUtils.dms_to_decimal(gps_longitude, gps_longitude_ref)        
        gps_obj = GPS(latitude=latitude, longitude=longitude, zone=0)
        zone = self.zone_dictionary.get_zone(gps_obj)
        
        return GPS(latitude=latitude, longitude=longitude, zone=zone)
    
    def _get_pm_data(self, gps_data: GPS) -> list[DataSensor]:
        """
        Obtiene los datos de material particulado de la zona estipulada usando la información de SIATA.   

        Args:
            gps_data: Datos de la zona geográfica de la imagen

        Returns:
            list[float]: Datos del material particulado de la zona
        """
        siata_pm_data = self.siata_gateway.get_data_by_zone(gps_data.zone)
        purpleair_pm_data = self.purpleair_gateway.get_data_by_zone(gps_data.zone)
        list_pm_data = [siata_pm_data, purpleair_pm_data]
        return list_pm_data


    def _get_pm_quantitative_estimation(self, pm_data: list[DataSensor], feature_vector: list[float]) -> PMEstimation:
        """
        Obtiene la estimación de la cantidad de material particulado presente.

        Args:
            pm_data: Datos del material particulado de la zona
            feature_vector: Vector de características

        Returns:
            PMEstimation: Estimación de la cantidad de material particulado presente y confianza de la estimación.
        """
        pm_estimation = self.estimation_ml_model_gateway.estimate_pm(pm_data, feature_vector)
        return pm_estimation

    def _get_pm_qualitative_estimation(self, pm_estimation: PMEstimation) -> PMEstimation:
        """
        Obtiene la estimación cualitativa de la cantidad de material particulado presente.

        Args:
            pm_estimation: Estimación de la cantidad de material particulado presente

        Returns:
            PMEstimation: Estimación cualitativa de la cantidad de material particulado presente
        """
        pass