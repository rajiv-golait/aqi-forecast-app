import pandas as pd
from src.data.validator import DataValidator

def test_validator_basic():
    df = pd.DataFrame({
        'city': ['A', 'B'],
        'datetime': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'pm25': [10, 20],
        'pm10': [30, 40]
    })
    validator = DataValidator()
    assert validator.validate(df) is True 