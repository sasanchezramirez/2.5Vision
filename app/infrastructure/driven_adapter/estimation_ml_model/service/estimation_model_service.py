import logging
from typing import Final
import numpy as np
import joblib

from app.domain.gateway.estimation_ml_model_gateway import EstimationMLModelGateway
from app.domain.model.data_sensors import DataSensor
from app.domain.model.pm_estimation import PMEstimation

logger: Final[logging.Logger] = logging.getLogger("Estimation Model Service")

class EstimationModelService(EstimationMLModelGateway):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = joblib.load(self.model_path)

    def estimate_pm(self, pm_data: list[DataSensor], feature_vector: list[float]) -> PMEstimation:
        """
        Estima la cantidad de material particulado presente en la imagen.

        Args:
            pm_data: Lista de datos de sensores de material particulado.
            feature_vector: Vector de características.
        """ 
        features = np.array(feature_vector)
        prediction = self.model.predict(features)
        confidence = self.model.predict_proba(features)
        return PMEstimation(pm_estimation=prediction, pm_estimation_confidence=confidence)
