# AgriBrain — ML Crop Recommendation System

AgriBrain is a machine learning–powered decision-support web app that recommends
suitable crops based on soil nutrients (N, P, K), temperature, humidity, pH, and
rainfall. Alongside the top-3 ML predictions, an expert-system layer adds
agronomic guidance: nutrient alerts, fertilizer suggestions, and a rough yield
outlook for the recommended crop.

## Features

- **ML-based recommendation** — trained classifier (see `models/model_comparison.json`
  for the model comparison results) predicts the top 3 most suitable crops with
  confidence scores.
- **Expert system insights** — rule-based checks on nutrients, rainfall, pH, and
  temperature produce human-readable alerts, fertilizer suggestions, and a yield
  estimate for the recommended crop.
- **Metrics dashboard** — `/metrics` page shows how the candidate models compared
  during training (accuracy, precision, recall, F1).
- **Web UI** — simple Flask + HTML/CSS frontend for entering field conditions and
  viewing results.

## Project Structure

```
agribrain-crop-recommendation/
├── app.py                   # Flask app: routes for /, /predict, /metrics
├── config.py                 # Central paths + expert-system rule tables
├── requirements.txt
├── data/
│   └── Crop_recommendation.csv   # Training dataset
├── models/                   # Trained model, scalers, and comparison metrics
│   ├── best_model.pkl
│   ├── minmaxscaler.pkl
│   ├── standscaler.pkl
│   ├── label_dict.pkl
│   ├── model_comparison.csv
│   └── model_comparison.json
├── src/
│   ├── ml_model.py            # CropRecommender: loads model/scalers, predicts
│   ├── expert_system.py       # Rule-based agronomic insights
│   ├── train_model.py         # Trains, compares, tunes, and saves the model
│   └── utils/
│       ├── feature_engineering.py
│       └── logger.py
├── templates/                 # Jinja2 templates (index, metrics)
├── static/                    # CSS + images
├── notebooks/
│   └── crop_analysis.ipynb    # Original exploratory analysis notebook
├── tests/
│   └── test_app.py
└── docs/
    └── methodology.md         # Notes on the modeling & expert-system approach
```

## Getting Started

### 1. Clone and set up the environment

```bash
git clone https://github.com/muzammil-code709/agribrain-ml-crop-recommendation-system.git
cd agribrain-ml-crop-recommendation-system
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the app

Pre-trained model artifacts are already included under `models/`, so you can run
the app directly:

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

### 3. (Optional) Retrain the model

To regenerate `models/best_model.pkl` and the scalers from `data/Crop_recommendation.csv`:

```bash
python -m src.train_model
```

This trains and cross-validates several candidate models (Random Forest, Decision
Tree, KNN, SVM, Gradient Boosting, and XGBoost if installed), tunes the best one,
and writes the model, scalers, label mapping, and comparison metrics into `models/`.

### 4. Run tests

```bash
pytest
```

## Tech Stack

- **Python** — application and modeling logic
- **Flask** — web framework
- **scikit-learn** — model training, scaling, evaluation
- **pandas / numpy** — data handling
- **xgboost** *(optional)* — included as a candidate model in training

## License

Released under the MIT License — see [LICENSE](LICENSE).
