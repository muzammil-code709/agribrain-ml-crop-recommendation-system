# AgriBrain — ML Crop Recommendation System

**AgriBrain** is a machine learning–powered agricultural decision-support system that recommends suitable crops based on key environmental and soil conditions.

The system combines a trained machine learning classifier with a rule-based **expert system** to provide both crop recommendations and interpretable agronomic guidance. Users provide soil and environmental parameters including **Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, and rainfall**.

The application returns the **top three recommended crops with confidence scores**, together with nutrient alerts, fertilizer suggestions, environmental observations, and a rough yield outlook.

---

## Overview

Selecting an appropriate crop depends on multiple interacting soil and environmental conditions. AgriBrain addresses this problem by combining:

* **Machine learning** for data-driven crop recommendation.
* **Feature preprocessing and scaling** for consistent model input.
* **An expert-system layer** for interpretable agronomic recommendations.
* **A Flask web application** for interactive predictions.
* **A model comparison pipeline** for evaluating multiple candidate algorithms.
* **A metrics dashboard** for inspecting model performance.

The project is designed as an end-to-end machine learning application covering data preparation, model training, evaluation, deployment, and user interaction.

---

## Key Features

### 🌱 Machine Learning Crop Recommendation

The trained classifier predicts the most suitable crops based on:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

The system provides the **top three predictions** along with their confidence scores.

Model comparison results are stored in:

```text
models/model_comparison.csv
models/model_comparison.json
```

### 🧠 Expert System

A rule-based expert system complements the machine learning prediction.

It analyzes the supplied field conditions and provides:

* Nutrient deficiency or excess alerts
* Fertilizer recommendations
* pH-related observations
* Temperature and rainfall observations
* Agronomic guidance
* Rough yield outlook for the recommended crop

This hybrid approach combines **predictive modeling** with **interpretable rule-based reasoning**.

### 📊 Model Evaluation Dashboard

The `/metrics` endpoint provides a web-based view of candidate model performance, including:

* Accuracy
* Precision
* Recall
* F1-score

The training pipeline compares multiple machine learning algorithms before selecting and tuning the best-performing candidate.

### 🌐 Web Application

AgriBrain provides a Flask-based web interface where users can:

1. Enter soil and environmental conditions.
2. Submit the field parameters.
3. Receive crop recommendations.
4. View confidence scores.
5. Review expert-system insights.
6. Inspect model evaluation metrics.

---

## System Architecture

```text
                  User
                   │
                   ▼
            Flask Web Interface
                   │
                   ▼
          Input Validation Layer
                   │
                   ▼
        Feature Preprocessing
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ML Recommendation   Expert System
          │                 │
          │                 ├── Nutrient Analysis
          │                 ├── Fertilizer Advice
          │                 ├── Environmental Alerts
          │                 └── Yield Outlook
          │
          ▼
      Top-3 Crops
    + Confidence Scores
          │
          └────────┬────────┘
                   ▼
            Final Results
                   │
                   ▼
                User
```

---

## Project Structure

```text
agribrain-crop-recommendation/
│
├── app.py                         # Flask application and web routes
├── config.py                      # Central configuration and rule tables
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
│
├── data/
│   └── Crop_recommendation.csv    # Crop recommendation dataset
│
├── models/
│   ├── best_model.pkl             # Selected trained model
│   ├── minmaxscaler.pkl           # Min-Max scaler
│   ├── standscaler.pkl            # Standard scaler
│   ├── label_dict.pkl             # Label/class mapping
│   ├── model_comparison.csv       # Model evaluation results
│   └── model_comparison.json      # Model evaluation results in JSON
│
├── src/
│   ├── __init__.py
│   ├── ml_model.py                # Model loading and prediction logic
│   ├── expert_system.py           # Rule-based agronomic reasoning
│   ├── train_model.py             # Training, comparison, tuning, and saving
│   │
│   └── utils/
│       ├── __init__.py
│       ├── feature_engineering.py # Feature preparation utilities
│       └── logger.py              # Application/training logging
│
├── templates/
│   ├── index.html                 # Main prediction interface
│   └── metrics.html               # Model metrics dashboard
│
├── static/
│   ├── css/
│   │   └── style.css              # Application styling
│   └── images/
│       └── bg.png                 # Application background
│
├── notebooks/
│   └── crop_analysis.ipynb        # Exploratory data analysis
│
├── tests/
│   └── test_app.py                # Application tests
│
└── docs/
    └── methodology.md             # Modeling and expert-system methodology
```

---

## Machine Learning Pipeline

The training pipeline follows these general stages:

```text
Dataset
   │
   ▼
Data Preparation
   │
   ▼
Feature Engineering
   │
   ▼
Train/Test Processing
   │
   ▼
Feature Scaling
   │
   ▼
Candidate Model Training
   │
   ├── Random Forest
   ├── Decision Tree
   ├── KNN
   ├── SVM
   ├── Gradient Boosting
   └── XGBoost (optional)
   │
   ▼
Model Evaluation
   │
   ▼
Best Model Selection
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Model + Scalers + Label Mapping
   │
   ▼
Saved to models/
```

The training pipeline evaluates candidate models using standard classification metrics and stores the comparison results for further analysis.

---

## Input Features

AgriBrain uses seven primary input variables:

| Feature       | Description                |
| ------------- | -------------------------- |
| `N`           | Nitrogen content in soil   |
| `P`           | Phosphorus content in soil |
| `K`           | Potassium content in soil  |
| `Temperature` | Environmental temperature  |
| `Humidity`    | Environmental humidity     |
| `pH`          | Soil acidity/alkalinity    |
| `Rainfall`    | Rainfall measurement       |

These features are processed using the project's saved preprocessing artifacts before being passed to the trained model.

---

## Output

For each valid input, AgriBrain provides:

### Machine Learning Output

* Top-3 recommended crops
* Prediction confidence scores
* Best-ranked crop

### Expert-System Output

* Nutrient condition analysis
* Nutrient alerts
* Fertilizer suggestions
* Environmental observations
* Approximate yield outlook

This provides users with both a **prediction** and an **explanation-oriented recommendation layer**.

---

# Getting Started

## Prerequisites

Make sure you have:

* Python 3.10+ recommended
* Git
* pip
* A modern web browser

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/muzammil-code709/agribrain-ml-crop-recommendation-system.git
```

Navigate to the project:

```bash
cd agribrain-ml-crop-recommendation-system
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Pre-trained model artifacts are included in the `models/` directory, so the application can be launched without retraining the model.

Run:

```bash
python app.py
```

The application will be available at:

```text
http://localhost:5000
```

Open the URL in your browser and enter the required soil and environmental parameters.

---

# Model Retraining

To retrain the machine learning models using the dataset in:

```text
data/Crop_recommendation.csv
```

run:

```bash
python -m src.train_model
```

The training pipeline:

1. Loads the dataset.
2. Prepares the features and target labels.
3. Performs feature preprocessing.
4. Trains multiple candidate models.
5. Evaluates model performance.
6. Selects the strongest candidate.
7. Performs model tuning.
8. Saves the trained model.
9. Saves preprocessing scalers.
10. Saves the label mapping.
11. Generates model comparison reports.

Generated artifacts are stored in:

```text
models/
```

---

# Running Tests

Run the test suite with:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

---

# Model Comparison

AgriBrain's training pipeline can evaluate several candidate classifiers, including:

* Random Forest
* Decision Tree
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gradient Boosting
* XGBoost, when available

The comparison results are saved as:

```text
models/model_comparison.csv
models/model_comparison.json
```

The evaluation considers standard classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score

---

# Technology Stack

### Programming Language

* **Python**

### Web Framework

* **Flask**
* **Jinja2**
* **HTML5**
* **CSS3**

### Machine Learning

* **scikit-learn**
* **XGBoost** *(optional)*

### Data Processing

* **pandas**
* **NumPy**

### Development & Testing

* **pytest**
* **Jupyter Notebook**

### Model Persistence

* **Joblib / Pickle-based model artifacts**

---

# Project Methodology

AgriBrain follows a hybrid decision-support approach.

### Machine Learning Layer

The machine learning layer learns relationships between soil/environmental conditions and suitable crops from historical crop recommendation data.

### Expert-System Layer

The expert system applies predefined agronomic rules to provide interpretable recommendations that complement the statistical model.

### Hybrid Decision Support

The final system combines both components:

```text
Machine Learning
      +
Expert-System Reasoning
      ↓
Crop Recommendation
+
Agronomic Guidance
```

This design allows the application to provide more than a simple classification result by adding rule-based context around the prediction.

For additional methodological details, see:

```text
docs/methodology.md
```

---

# API / Application Routes

The Flask application currently provides the following primary routes:

| Route      | Method | Purpose                                                  |
| ---------- | ------ | -------------------------------------------------------- |
| `/`        | `GET`  | Displays the crop recommendation interface               |
| `/predict` | `POST` | Processes field conditions and generates recommendations |
| `/metrics` | `GET`  | Displays model comparison metrics                        |

---

# Data

The project uses a crop recommendation dataset containing soil and environmental characteristics associated with crop classes.

The primary dataset is located at:

```text
data/Crop_recommendation.csv
```

The dataset is used during model training and evaluation.

---

# Reproducibility

The repository includes the trained model and preprocessing artifacts required to run the application without retraining:

```text
models/
├── best_model.pkl
├── minmaxscaler.pkl
├── standscaler.pkl
└── label_dict.pkl
```

This allows users to clone the repository, install the dependencies, and run the application directly.

For reproducible retraining, use the provided dataset and training pipeline:

```bash
python -m src.train_model
```

---

# Limitations

AgriBrain is intended as a **decision-support and educational machine learning application**, not as a replacement for professional agricultural consultation.

The quality of recommendations depends on:

* Dataset quality and representativeness
* Model performance
* Accuracy of user-provided field measurements
* Coverage of the training data
* Validity of predefined expert-system rules

The yield outlook and agronomic guidance should therefore be treated as approximate recommendations rather than guaranteed outcomes.

---

# Future Improvements

Potential future improvements include:

* Integration with real-time weather APIs
* Location-aware recommendations
* Soil sensor integration
* Larger and more geographically diverse datasets
* Explainable AI techniques such as SHAP
* Improved yield prediction using dedicated regression models
* Database-backed user and field management
* Model monitoring and versioning
* Cloud deployment
* REST API support
* Mobile-friendly interface
* Automated model retraining pipelines

---

# Repository Workflow

The recommended development workflow is:

```text
Data
  ↓
Exploration
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Selection
  ↓
Model Persistence
  ↓
Flask Integration
  ↓
Expert-System Reasoning
  ↓
Testing
  ↓
Deployment
```

---

# License

This project is released under the **MIT License**.

See the [LICENSE](LICENSE) file for the complete license text.

---

# Author

**Muzammil**

GitHub: [@muzammil-code709](https://github.com/muzammil-code709)

---

## Project Status

**Status:** Active Development

AgriBrain is currently being developed as an end-to-end machine learning and agricultural decision-support project, with ongoing improvements to its modeling pipeline, software architecture, testing, and documentation.
