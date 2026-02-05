
# Model loading and prediction logic

import joblib
import numpy as np
import pandas as pd
from typing import Dict
import logging
import xgboost as xgb

logger = logging.getLogger(__name__)

class HousePricePredictor:
    
    def __init__(self, model_path: str = "models/xgboost_model.plk"):
        
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {e}")
            raise
        
    def predict(self, features: pd.DataFrame) -> float:
        
        try:
            prediction = self.model.predict(features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Prediction failed {e}")
            raise
        
    def predict_with_confidence(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        prediction = self.model.predict(features)

        booster = self.model.get_booster()
        num_trees = booster.num_boosted_rounds()

        dmatrix = xgb.DMatrix(features)

        tree_predictions = np.array([
            booster.predict(dmatrix, iteration_range=(i, i + 1))
            for i in range(num_trees)
        ])

        std = np.std(tree_predictions, axis=0)

        return {
            "prediction": prediction,
            "lower_bound": prediction - 1.96 * std,
            "upper_bound": prediction + 1.96 * std
        }
