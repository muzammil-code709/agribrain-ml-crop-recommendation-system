# ml_model.py

import pickle
import numpy as np
from config import MODEL_PATHS
from src.utils.feature_engineering import engineer_features

class CropRecommender:
    def __init__(self):
        self.model = None
        self.sc = None
        self.ms = None
        self.label_dict = None
        self.load_error = None
        self._load_models()

    def _load_models(self):
        try:
            with MODEL_PATHS["model"].open("rb") as f:
                self.model = pickle.load(f)
            with MODEL_PATHS["standard_scaler"].open("rb") as f:
                self.sc = pickle.load(f)
            with MODEL_PATHS["minmax_scaler"].open("rb") as f:
                self.ms = pickle.load(f)
            with MODEL_PATHS["label_dict"].open("rb") as f:
                self.label_dict = pickle.load(f)
        except Exception as exc:
            self.load_error = f"Failed to load AI model or scalers: {exc}"
            print(self.load_error)

    def predict(self, feature_list):
        """
        Returns top 3 predictions and their confidence scores.
        feature_list: [N, P, K, temperature, humidity, ph, rainfall]
        """
        if self.load_error:
            raise RuntimeError(self.load_error)

        # Apply Feature Engineering
        # feature_list has 7 base features.
        engineered_vector = engineer_features(*feature_list)
        
        # Scale features
        feature_vector_2d = np.array([engineered_vector])
        scaled_features = self.ms.transform(feature_vector_2d)
        final_features = self.sc.transform(scaled_features)

        try:
            # Try to get probabilities
            probabilities = self.model.predict_proba(final_features)[0]
            
            # Get top 3 indices
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            
            top_3_predictions = []
            for idx in top_3_indices:
                crop_name = self.label_dict.get(idx, "Unknown")
                confidence = round(probabilities[idx] * 100, 2)
                top_3_predictions.append({"crop": crop_name, "confidence": confidence})
                
            return top_3_predictions
        
        except AttributeError:
            # Fallback if model doesn't support predict_proba
            prediction_idx = self.model.predict(final_features)[0]
            crop_name = self.label_dict.get(prediction_idx, "Unknown")
            return [{"crop": crop_name, "confidence": 100.0}]

# Initialize single instance
recommender = CropRecommender()
