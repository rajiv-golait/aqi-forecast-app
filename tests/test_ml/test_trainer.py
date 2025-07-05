import pandas as pd
from src.ml.trainer import AQIModelTrainer
 
def test_trainer_init():
    trainer = AQIModelTrainer()
    assert trainer.model is None
    assert trainer.scaler is not None 