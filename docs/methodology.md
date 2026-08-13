# Methodology

This document describes how AgriBrain turns raw soil/climate readings into a
crop recommendation, and how the underlying model was built.

## 1. Input Features

The user supplies 7 raw measurements for a field:

| Feature | Description |
|---|---|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| Temperature | Average temperature (°C) |
| Humidity | Relative humidity (%) |
| pH | Soil pH |
| Rainfall | Rainfall (mm) |

## 2. Feature Engineering

`src/utils/feature_engineering.py` derives 4 additional features from the raw
7, giving the model more informative signal on nutrient balance and climate
interaction:

- `N/P` ratio
- `N/K` ratio
- `P/K` ratio
- `temperature * humidity` (a simple heat-index proxy)

The same transformation (`engineer_features_df`) is applied during training and
(`engineer_features`) at inference time, so the feature space the model sees is
identical in both cases.

## 3. Preprocessing

Engineered features are scaled in two steps before being passed to the model:

1. `MinMaxScaler` — normalizes values into a common range.
2. `StandardScaler` — standardizes to zero mean / unit variance.

Both scalers are fit on the training split only, then persisted (`minmaxscaler.pkl`,
`standscaler.pkl`) so the exact same transformation is reproduced at inference
time in `src/ml_model.py`.

## 4. Model Selection

`src/train_model.py` trains and 5-fold cross-validates several candidate
classifiers on the engineered/scaled features:

- Random Forest
- Decision Tree
- K-Nearest Neighbors
- Support Vector Machine
- Gradient Boosting
- XGBoost (if installed)

For each candidate it records cross-validation accuracy, held-out test
accuracy, precision, recall, and F1 (weighted). Results are saved to
`models/model_comparison.csv` / `.json` and surfaced in the app's `/metrics`
page.

The candidate with the best cross-validation accuracy is then tuned further
with `GridSearchCV` (hyperparameter search over depth, estimator count, etc.,
depending on the model type) before being saved as `models/best_model.pkl`.

## 5. Inference

At request time, `CropRecommender` (`src/ml_model.py`):

1. Loads the trained model, both scalers, and the label mapping once at
   startup.
2. Applies the same feature engineering + scaling pipeline to the incoming
   7 values.
3. Calls `predict_proba` (falling back to a plain `predict` if the underlying
   model doesn't support probabilities) and returns the top 3 crops by
   confidence.

## 6. Expert System Layer

Independently of the ML prediction, `src/expert_system.py` applies simple
threshold-based agronomic rules (defined in `config.py` per crop, with a
generic fallback) to the same raw inputs. This produces:

- **Alerts** — out-of-range nutrients, rainfall, pH, or temperature for the
  recommended crop.
- **Fertilizer suggestions** — targeted recommendations tied to whichever
  nutrient(s) are deficient.
- **Guidelines** — general crop-specific management notes.
- **Yield estimate** — a coarse qualitative estimate derived from how many
  alerts were triggered.

This layer is intentionally simple and rule-based (not learned) so its
reasoning is transparent and easy to extend with agronomic domain knowledge
independently of the ML model.
