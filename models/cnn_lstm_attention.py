import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class CNNLSTMAttentionModel:
    def __init__(self, n_features, sequence_length=168, forecast_horizon=72):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = self._build_model()
    
    def _build_model(self):
        """Build CNN-LSTM with Attention model"""
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # CNN layers for feature extraction
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # LSTM layers
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers for forecasting
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output layer - predict PM2.5 and PM10 for each hour in forecast horizon
        outputs = layers.Dense(self.forecast_horizon * 2)(x)  # 2 for PM2.5 and PM10
        outputs = layers.Reshape((self.forecast_horizon, 2))(outputs)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, data, target_cols=['pm25', 'pm10']):
        """Prepare sequences for training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon][target_cols]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)