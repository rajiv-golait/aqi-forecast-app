"""Model evaluator for AQI forecasting models."""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class AQIModelEvaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate model predictions using MAE and RMSE."""
        results = {}
        if y_true.shape != y_pred.shape:
            logger.error(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
            return results
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        results['mae'] = mae
        results['rmse'] = rmse
        logger.info(f"Evaluation results: MAE={mae:.4f}, RMSE={rmse:.4f}")
        return results 