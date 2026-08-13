import sys
import time
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Allow running this file directly (`python src/train_model.py`) as well as
# as a module (`python -m src.train_model`) by ensuring the project root is
# importable either way.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PATH, MODELS_DIR, MODEL_PATHS, MODEL_COMPARISON_PATH, MODEL_COMPARISON_CSV_PATH
from src.utils.feature_engineering import engineer_features_df

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Lowercase crop labels for consistency
df['label'] = df['label'].str.lower()

X = df.drop("label", axis=1)
y = df["label"]

# Apply feature engineering
X = engineer_features_df(X)

# Map labels to integers
crop_labels = sorted(y.unique())
label_to_id = {label: idx for idx, label in enumerate(crop_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
y_mapped = y.map(label_to_id)

print(f"Dataset loaded. Total records: {len(df)}")
print(f"Total crop classes: {len(crop_labels)}")
print(f"Features used ({len(X.columns)}): {list(X.columns)}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped)

# Preprocessing
print("Applying scalers...")
ms = MinMaxScaler()
X_train_ms = ms.fit_transform(X_train)
X_test_ms = ms.transform(X_test)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_ms)
X_test_sc = sc.transform(X_test_ms)

# Define models for initial comparison
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42) # reduced n_estimators for speed
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

results = []

print("Training and evaluating models with Cross-Validation...")
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Train on full training set for test evaluation
    model.fit(X_train_sc, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_sc)
    
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cv_scores.mean()
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results.append({
        "Model": name,
        "CV Accuracy": round(cv_acc, 4),
        "Test Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "Training Time (s)": round(train_time, 4)
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Identify best model based on CV Accuracy
best_idx = results_df['CV Accuracy'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
print(f"\nBest model found: {best_model_name}")

print("Performing Hyperparameter Tuning on best model...")
tuned_model = None
if best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train_sc, y_train)
    tuned_model = grid.best_estimator_
elif best_model_name == "XGBoost":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train_sc, y_train)
    tuned_model = grid.best_estimator_
else:
    # If it's another model, just use the trained instance (tuning others can be added if needed)
    tuned_model = models[best_model_name]

# Final Evaluation of Tuned Model
y_pred_final = tuned_model.predict(X_test_sc)
final_acc = accuracy_score(y_test, y_pred_final)
print(f"\nTuned Model Test Accuracy: {final_acc:.4f}")

# Save results to CSV and JSON for frontend
results_df.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)
results_dict = results_df.to_dict(orient='records')
with open(MODEL_COMPARISON_PATH, "w") as f:
    json.dump(results_dict, f, indent=4)
print("Saved model comparison metrics.")

# Save the best model and scalers
print("Saving best tuned model and scalers...")
with MODEL_PATHS["model"].open("wb") as f:
    pickle.dump(tuned_model, f)
with MODEL_PATHS["standard_scaler"].open("wb") as f:
    pickle.dump(sc, f)
with MODEL_PATHS["minmax_scaler"].open("wb") as f:
    pickle.dump(ms, f)
with MODEL_PATHS["label_dict"].open("wb") as f:
    pickle.dump(id_to_label, f)

print("Training pipeline completed successfully.")
