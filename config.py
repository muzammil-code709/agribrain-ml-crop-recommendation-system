# config.py

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "Crop_recommendation.csv"

MODEL_PATHS = {
    "model": MODELS_DIR / "best_model.pkl",
    "standard_scaler": MODELS_DIR / "standscaler.pkl",
    "minmax_scaler": MODELS_DIR / "minmaxscaler.pkl",
    "label_dict": MODELS_DIR / "label_dict.pkl",
}

MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
MODEL_COMPARISON_CSV_PATH = MODELS_DIR / "model_comparison.csv"

CROP_RULES = {
    "Rice": {
        "nutrients": {"N": (90, 130), "P": (35, 60), "K": (40, 70)},
        "rainfall": (100, 300),
        "ph": (5.5, 6.5),
        "notes": "Apply urea and green manure for nitrogen, and monitor standing water carefully.",
    },
    "Maize": {
        "nutrients": {"N": (80, 120), "P": (35, 55), "K": (35, 55)},
        "rainfall": (50, 150),
        "ph": (5.5, 7.0),
        "notes": "Keep soil moist with split nitrogen applications and avoid waterlogging.",
    },
    "Cotton": {
        "nutrients": {"N": (60, 90), "P": (40, 60), "K": (40, 70)},
        "rainfall": (50, 120),
        "ph": (6.0, 7.5),
        "notes": "Use potassium-rich fertilizers for fiber development and conserve soil moisture.",
    },
    "Mango": {
        "nutrients": {"N": (70, 100), "P": (45, 65), "K": (50, 80)},
        "rainfall": (75, 150),
        "ph": (5.5, 7.5),
        "notes": "Ensure good drainage and add organic compost to improve soil structure.",
    },
    "Apple": {
        "nutrients": {"N": (40, 70), "P": (30, 55), "K": (50, 80)},
        "rainfall": (75, 125),
        "ph": (6.0, 7.0),
        "notes": "Use mulching and maintain balanced soil moisture during flowering.",
    },
}

DEFAULT_RULES = {
    "nutrients": {"N": (50, 100), "P": (30, 60), "K": (30, 70)},
    "rainfall": (40, 140),
    "ph": (6.0, 7.0),
    "notes": "Check general nutrient balance and adjust irrigation to protect crop health.",
}
