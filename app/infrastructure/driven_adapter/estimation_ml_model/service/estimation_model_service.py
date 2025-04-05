import logging
from typing import Final
import numpy as np
import joblib
import os

from app.domain.gateway.estimation_ml_model_gateway import EstimationMLModelGateway
from app.domain.model.data_sensors import DataSensor
from app.domain.model.pm_estimation import PMEstimation

logger: Final[logging.Logger] = logging.getLogger("Estimation Model Service")

class EstimationModelService(EstimationMLModelGateway):
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        # No cargamos el modelo en la inicialización
        logger.info("Servicio de estimación inicializado sin modelo")

    def estimate_pm(self, pm_data: list[DataSensor], feature_vector: list[float]) -> PMEstimation:
        """
        Estima la cantidad de material particulado presente en la imagen.
        Como no tenemos el modelo entrenado aún, devolvemos valores simulados.

        Args:
            pm_data: Lista de datos de sensores de material particulado.
            feature_vector: Vector de características.
        """ 
        logger.info("Usando valores simulados para estimación PM ya que no tenemos el modelo aún")
        # Valores simulados para pruebas
        prediction = 25.0  # Valor PM2.5 simulado
        confidence = 0.85  # Confianza simulada
        
        return PMEstimation(
            pm_estimation=prediction,
            pm_estimation_confidence=confidence,
            pm_qualitative_estimation="Moderado"  # Valor cualitativo simulado
        )
