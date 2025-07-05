import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EnhancedCNNLSTMModel:
    def __init__(self, sequence_length, n_features, forecast_hours):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.forecast_hours = forecast_hours
        self.model = self._build_model()
        
    def _build_model(self):
        """Build enhanced CNN-LSTM model with attention"""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # CNN layers for feature extraction
        cnn_out = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        cnn_out = layers.BatchNormalization()(cnn_out)
        cnn_out = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(cnn_out)
        cnn_out = layers.BatchNormalization()(cnn_out)
        cnn_out = layers.MaxPooling1D(pool_size=2)(cnn_out)
        
        # Bidirectional LSTM layers
        lstm_out = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(cnn_out)
        lstm_out = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(lstm_out)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=64,
            dropout=0.2
        )(lstm_out, lstm_out)
        
        # Combine attention with LSTM output
        combined = layers.Add()([lstm_out, attention])
        combined = layers.LayerNormalization()(combined)
        
        # Global context
        global_context = layers.GlobalAveragePooling1D()(combined)
        
        # Dense layers for prediction
        dense_out = layers.Dense(256, activation='relu')(global_context)
        dense_out = layers.Dropout(0.3)(dense_out)
        dense_out = layers.Dense(128, activation='relu')(dense_out)
        dense_out = layers.Dropout(0.3)(dense_out)
        
        # Output layer
        outputs = layers.Dense(self.forecast_hours)(dense_out)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._weighted_mae,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _weighted_mae(self, y_true, y_pred):
        """Custom loss function with higher weight for extreme AQI values"""
        # Calculate weights based on AQI severity
        weights = tf.where(y_true > 300, 2.0,  # Severe
                  tf.where(y_true > 200, 1.5,  # Very Poor
                  tf.where(y_true > 100, 1.2,  # Poor
                  1.0)))  # Good/Moderate
        
        mae = tf.abs(y_true - y_pred)
        weighted_mae = mae * weights
        
        return tf.reduce_mean(weighted_mae)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load model"""
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'_weighted_mae': self._weighted_mae}
        )